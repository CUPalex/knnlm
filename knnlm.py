import os

import logging
import time
import numpy as np
import torch
from torch import nn
from enum import Enum, auto
from pathlib import Path

import pickle

import faiss
import faiss.contrib.torch_utils

logger = logging.getLogger(__name__)
logger.setLevel(20)

class DIST(Enum):
    l2 = auto()
    dot = auto()

    @staticmethod
    def from_string(s):
        try:
            return DIST[s.lower()]
        except KeyError:
            raise ValueError()


class KEY_TYPE(Enum):
    last_ffn_input = auto()
    last_ffn_output = auto()

    @staticmethod
    def from_string(s):
        try:
            return KEY_TYPE[s.lower()]
        except KeyError:
            raise ValueError()

class KNNWrapper(object):
    def __init__(self, dstore_size, dstore_dir, dimension, 
            knn_sim_func=None, knn_keytype=None,
            no_load_keys=False, move_dstore_to_mem=False, knn_gpu=True,
            recompute_dists = False,
            k=1024, lmbda=0.25, knn_temp=1.0, probe=32, save_eval_data=False,
            save_eval_data_path=None):
        self.dstore_size = dstore_size
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.lmbda = lmbda
        self.k = k
        self.knn_temperature = knn_temp
        self.probe = probe
        self.knn_sim_func = DIST.l2 if knn_sim_func is None else knn_sim_func
        self.knn_keytype = KEY_TYPE.last_ffn_input if knn_keytype is None else knn_keytype
        self.no_load_keys = no_load_keys
        self.recompute_dists = recompute_dists
        self.move_dstore_to_mem = move_dstore_to_mem
        self.knn_gpu = knn_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prompt_input_ids = None
        self.keys = None
        self.values = None
        self.prompt_attention_mask = None
        self.model = None
        self.vocab_size = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.hook_handles = []

        self.save_eval_data = save_eval_data
        if self.save_eval_data:
            self.save_eval_data_path = save_eval_data_path
            self.what_to_save = {}

        dist_type_to_dist_func = {
            DIST.l2: KNNWrapper.l2,
            DIST.dot: KNNWrapper.dotprod,
        }
        self.dist_func = dist_type_to_dist_func[knn_sim_func] # l2 or dot product function


    def setup_faiss(self):
        if not self.dstore_dir:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()
        index_name = get_index_path(self.dstore_dir, self.model.config.model_type, self.dstore_size, self.dimension) 
        cpu_index = faiss.read_index(index_name, faiss.IO_FLAG_ONDISK_SAME_DIR)
        logger.info(f'Reading datastore took {time.time() - start} s')
        cpu_index.nprobe = self.probe

        if self.knn_gpu:
            start = time.time()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index, co)
            logger.info(f'Moving index to GPU took {time.time() - start} s')
        else:
            gpu_index = cpu_index

        # make_direct_map() allows calling reconstruct(n), 
        # and reconstructing key vectors given their ids
        # currently, this is implemented only for CPU indexes:
        # https://github.com/facebookresearch/faiss/issues/2181
        cpu_index.make_direct_map()

        keys_vals_prefix = get_dstore_path(self.dstore_dir, self.model.config.model_type, self.dstore_size, self.dimension)
        if not self.no_load_keys:
            self.keys = np.memmap(f'{keys_vals_prefix}_keys.npy', dtype=np.float16, mode='r',
                                  shape=(self.dstore_size, self.dimension))
        self.vals = np.memmap(f'{keys_vals_prefix}_vals.npy', dtype=np.int32, mode='r',
                              shape=(self.dstore_size, 1))
        # self.vals = torch.from_numpy(self.vals).to(self.device)

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if self.move_dstore_to_mem:
            logger.info('Loading to memory...')
            start = time.time()

            if not self.no_load_keys:
                del self.keys
                self.keys_from_memmap = np.memmap(f'{keys_vals_prefix}_keys.npy', 
                    dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))
                self.keys = self.keys_from_memmap[:].astype(np.float16)

            del self.vals
            vals_from_memmap = np.memmap(f'{keys_vals_prefix}_vals.npy', dtype=np.int32, mode='r', shape=(self.dstore_size, 1))
            self.vals = torch.from_numpy(vals_from_memmap[:]).long().to(self.device)
            del vals_from_memmap
            logger.info('Loading to memory took {} s'.format(time.time() - start))

        return cpu_index, gpu_index

    def break_into(self, model):
        self.model = model
        model.broken_into = True
        self.reconstruct_index, self.index = self.setup_faiss()
        self.is_encoder_decoder = model.config.is_encoder_decoder

        # Inject our pre_forward_hook to capture the labels at every forward pass
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook
        
        # Inject our activation_capturer to capture the activations at every forward pass
        layer_to_capture_fn, capture_input = KNNWrapper.model_layer_to_capture[model.config.model_type][self.knn_keytype]
        layer_to_capture = layer_to_capture_fn(model)
        self.activation_capturer = ActivationCapturer(layer_to_capture, capture_input=capture_input)
        self.register_hook(layer_to_capture, self.activation_capturer)

        # Inject our main function after the model's final layer
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)
        self.vocab_size = final_layer.out_features

    def get_knns(self, queries):
        if not self.knn_gpu:
            queries = queries.cpu()
        dists, knns = self.index.search(queries, self.k)
        dists, knns = dists.to(self.device), knns.to(self.device)
        return dists, knns

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        self.labels = labels
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)

    def post_forward_hook(self, module, input, output):
        batch, time_dim, vocab_size = output.shape
        shift = 0 if self.is_encoder_decoder else 1
        lm_logits = output
        lm_logits = torch.nn.functional.log_softmax(lm_logits, dim=-1) # (batch, time, vocab)
        queries = self.activation_capturer.captured # (batch, time, dim)

        if self.labels is None:
            nonpad_mask = torch.cat([
                torch.zeros([batch, time_dim - 1], dtype=torch.bool),
                torch.ones([batch, 1], dtype=torch.bool),
            ], axis=-1).to(self.device)
        else:
            nonpad_mask = torch.cat([
                self.labels[:, shift:] != -100, 
                torch.zeros([self.labels.shape[0], shift], dtype=torch.bool).to(self.device)
            ], axis=-1)

        lm_logits = lm_logits[nonpad_mask]
        queries = queries[nonpad_mask] # (nonpad, dim)
        
        dists, knns = self.get_knns(queries) # (nonpad batch * time, k)
        if self.recompute_dists:
            knns_vecs = torch.from_numpy(self.keys[knns]).to(self.device)
            dists = self.dist_func(queries, knns_vecs) 
        
        neg_dists = -dists
        knn_log_probs, _ = self.knns_to_log_prob(knns, neg_dists)
        
        interpolated_scores = KNNWrapper.interpolate(knn_log_probs, lm_logits, self.lmbda) # (nonpad, vocab)

        if self.save_eval_data:
            to_save_in_iter = dict(
                lm_log_probs=lm_logits.squeeze(0).to("cpu"),
                knn_log_probs=knn_log_probs.to("cpu"),
                dists=dists.to("cpu"),
                knns=knns.to("cpu"),
                labels=self.labels[:, shift:][self.labels[:, shift:] != -100].to("cpu"),
                queries=queries.squeeze(0).to("cpu"),
                interpolated_scores=interpolated_scores.to("cpu")
            )

            if self.what_to_save == {}:
                for key in to_save_in_iter:
                    self.what_to_save[key] = to_save_in_iter[key]
                    if key != "labels":
                        self.what_to_save[key] = self.what_to_save[key].to(torch.float32)
            else:
                for key in to_save_in_iter:
                    self.what_to_save[key] = torch.cat([self.what_to_save[key], to_save_in_iter[key]], dim=0)
                    if key != "labels":
                        self.what_to_save[key] = self.what_to_save[key].to(torch.float32)
            

        output[nonpad_mask] = interpolated_scores
        return output 

    def knns_to_log_prob(self, knns, neg_dists):
        probs = torch.nn.functional.softmax(neg_dists / self.knn_temperature, dim=-1)
        vals_at_knns = self.vals[knns].squeeze(-1) # (nonpad batch * time, k)
        knn_log_probs = torch.full(size=(vals_at_knns.shape[:-1] + (self.vocab_size,)), fill_value=0.0).to(self.device) \
            .scatter_add(dim=-1, index=vals_at_knns, src=probs).log() # (nonpad_batch * time, vocab)
        knn_log_probs = torch.nan_to_num(knn_log_probs, nan=None, neginf=-10000.0)
        return knn_log_probs, vals_at_knns
        
    def register_hook(self, layer, func, pre=False):
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)

    def break_out(self):
        if self.save_eval_data:
            with open(self.save_eval_data_path, "wb") as file:
                pickle.dump(self.what_to_save, file)
        for h in self.hook_handles:
            h.remove()
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None
    
    def get_metrics(self):
        return {}
    
    @staticmethod
    def l2(query, keys):
        # query: (batch*time, dim)
        # keys:  (batch*time, k, dim)
        # returns: (batch*time, k)
        return torch.sum((query.unsqueeze(-2) - keys)**2, dim=-1)

    @staticmethod
    def dotprod(query, keys):
        # query: (batch, beams, dim)
        # keys:  (batch, 1, time, dim)
        # returns: (batch, beams, time)
        return torch.sum((query.unsqueeze(-2) * keys), dim=-1)


    @staticmethod
    def interpolate(knn_log_probs, lm_log_probs, lmbda):
        interpolated = torch.logaddexp(
            lm_log_probs + np.log(1 - lmbda), 
            knn_log_probs + np.log(lmbda))

        return interpolated

    @staticmethod
    def get_model_last_layer(model_type):
        # works for gpt2, marian, t5. If a model does not have an ".lm_head" layer, 
        # add an "if model_type is ..." statement here, and return the output embedding layer
        return lambda model: model.lm_head

    @staticmethod
    def get_model_embedding_layer(model_type):
        if model_type.startswith('gpt2'):
            return lambda model: model.transformer.wte

    # For every model name and key type, returns a lambda that returns the relevant layer in the model, 
    # and whether the input of that layer should be captured (True) or the output (False)
    model_layer_to_capture = {
        'bart': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.layers[-1].fc1, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.layers[-1], False),
        },
        'gpt2': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.h[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.h[-1], False),
        },
        'marian': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.layers[-1].fc1, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.layers[-1], False),
        },
        't5': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.block[-1].layer[2].DenseReluDense, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.block[-1].layer[2], False),
        }
}
    

class KNNSaver(object):
    def __init__(self, dstore_size, dstore_dir, dimension,
                 build_dstore, save_eval_data, build_index_on_the_go,
                 semem_thres, ncentroids, code_size, probe, num_keys_to_add_at_a_time,
                 knn_keytype=None):
        self.dstore_size = dstore_size
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.knn_keytype = KEY_TYPE.last_ffn_input if knn_keytype is None else knn_keytype

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.build_dstore = build_dstore
        self.build_index_on_the_go = build_index_on_the_go
        if self.build_dstore or self.build_index_on_the_go:
            self.cur_dstore_size = 0
            self.dstore_size = None
            self.combined_dataset = False

        self.save_eval_data = save_eval_data

        self.num_keys_to_add_at_a_time = num_keys_to_add_at_a_time
        if self.build_index_on_the_go:
            logger.info('Building index')
            # Initialize faiss index
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFPQ(quantizer, self.dimension,
                ncentroids, code_size, 8)
            self.index.nprobe = probe

            self.total_keys_added = 0

        if semem_thres is not None:
            self.semem_thres = semem_thres

        self.model = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.dstore_keys = None
        self.dstore_vals = None
        self.labels = None
        self.hook_handles = []

        logger.info(f'keytype being saved: {self.knn_keytype}')
        logger.info('Saving fp16')

    def break_into(self, model):
        self.model = model
        model.broken_into = True
        self.is_encoder_decoder = model.config.is_encoder_decoder
        
        # Inject our activation_capturer to capture the activations at every forward pass
        layer_to_capture_fn, capture_input = KNNWrapper.model_layer_to_capture[model.config.model_type][self.knn_keytype]
        layer_to_capture = layer_to_capture_fn(model)
        self.activation_capturer = ActivationCapturer(layer_to_capture, capture_input=capture_input)
        self.register_hook(layer_to_capture, self.activation_capturer)

        # Inject our pre_forward_hook to capture the labels at every forward pass
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook
        
        # Inject our main function after the model's final layer
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)

        if self.build_dstore or self.build_index_on_the_go:
            self.temp_save_dir_prefix = get_temp_dstore_path(self.dstore_dir, model.config.model_type, self.dimension)
            if os.path.exists(self.temp_save_dir_prefix):
                raise ValueError('Directory for saving dstore alredy exists. Please provide non-existent path.')
            else:
                Path(self.temp_save_dir_prefix).mkdir(parents=True, exist_ok=True)
        else:
            self.keys_vals_prefix = get_dstore_path(self.dstore_dir, model.config.model_type, self.dstore_size, self.dimension)
            keys_filename = f'{self.keys_vals_prefix}_keys.npy'
            vals_filename = f'{self.keys_vals_prefix}_vals.npy'
            if os.path.exists(keys_filename) and os.path.exists(vals_filename):
                self.dstore_keys = np.memmap(keys_filename, dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))
                self.dstore_vals = np.memmap(vals_filename, dtype=np.int32, mode='r', shape=(self.dstore_size, 1))
            else:
                raise ValueError(f'Could not read the provided datastore. Path to the datastore {keys_filename} or {vals_filename} does not exist.')
            
    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if labels is None:
            raise ValueError('labels must be provided when saving a datastore. Are you using --predict_with_generate by mistake? If so, disable it')
        self.labels = labels
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)

    def post_forward_hook(self, module, input, output):
        shift = 0 if self.is_encoder_decoder else 1
        captured_keys = self.activation_capturer.captured
        if shift == 1:
            captured_keys = captured_keys[:, :-shift]
        captured_keys = captured_keys.flatten(0, 1) # (batch * time, dim)
        captured_values = self.labels[:, shift:].flatten(0, 1) # (batch * time)

        nonpad_mask = captured_values != -100
        keys = captured_keys[nonpad_mask]
        values = captured_values[nonpad_mask]

        if self.save_eval_data or self.semem_thres is not None:
            lm_logits = torch.nn.functional.log_softmax(output, dim=-1)[:, shift:].flatten(0, 1) # (batch * time, vocab)
            lm_scores = lm_logits[torch.arange(lm_logits.shape[0]), captured_values] # (batch * time)

        if self.semem_thres is not None:
            lm_scores_without_pad = lm_scores[nonpad_mask]
            captured_keys = captured_keys[lm_scores_without_pad < self.semem_thres]
            captured_values = captured_values[lm_scores_without_pad < self.semem_thres]
            logger.info(f"Saving {sum(lm_scores_without_pad < self.semem_thres) / lm_scores_without_pad.shape[0]:.3f}")

        batch_time_size = keys.shape[0]
        if self.build_dstore:
            if batch_time_size == 0:
                logger.info("Nothing to write")
            else:
                keys_arr = np.memmap(f"{self.temp_save_dir_prefix}_temp_keys_{self.cur_dstore_size}_{batch_time_size}.npy",
                                dtype=np.float16, mode="w+",
                                shape=(batch_time_size, self.dimension))
                vals_arr = np.memmap(f"{self.temp_save_dir_prefix}_temp_vals_{self.cur_dstore_size}_{batch_time_size}.npy",
                            dtype=np.int32, mode="w+",
                            shape=(batch_time_size, 1))

                keys_arr[:,:] = keys.cpu().numpy().astype(np.float16)
                vals_arr[:,:] = values.unsqueeze(-1).cpu().numpy().astype(np.int32)

                del keys_arr
                del vals_arr

                self.cur_dstore_size += batch_time_size
                
        if self.build_index_on_the_go:
            num_batches = (batch_time_size // self.num_keys_to_add_at_a_time + \
                            int(batch_time_size % self.num_keys_to_add_at_a_time != 0))
            batches = [batch_time_size % self.num_keys_to_add_at_a_time]
            batches.extend([self.num_keys_to_add_at_a_time for _ in range(num_batches - 1)])

            print(batches, batch_time_size, self.num_keys_to_add_at_a_time,
                    self.cur_dstore_size, self.total_keys_added,
                    self.index.is_trained)
        
            for i, batch in enumerate(batches):
                keys_arr = np.memmap(f"{self.temp_save_dir_prefix}_temp_keys_{self.cur_dstore_size}_{batch}.npy",
                            dtype=np.float16, mode="w+",
                            shape=(batch, self.dimension))
                vals_arr = np.memmap(f"{self.temp_save_dir_prefix}_temp_vals_{self.cur_dstore_size}_{batch}.npy",
                            dtype=np.int32, mode="w+",
                            shape=(batch, 1))

                keys_arr[:,:] = keys[sum(batches[:i]):sum(batches[:i]) + batch, :].cpu().numpy().astype(np.float16)
                vals_arr[:,:] = values[sum(batches[:i]):sum(batches[:i]) + batch].unsqueeze(-1).cpu().numpy().astype(np.int32)

                del keys_arr
                del vals_arr

                self.cur_dstore_size += batch
                if self.cur_dstore_size - self.total_keys_added >= self.num_keys_to_add_at_a_time:
                    self.update_index()

        
        return output

    def combine_dataset(self):
        logger.info('Combining dataset...')
        self.dstore_size = self.cur_dstore_size
        self.keys_vals_prefix = get_dstore_path(self.dstore_dir, self.model.config.model_type, self.dstore_size, self.dimension)

        Path(self.keys_vals_prefix).parent.mkdir(parents=True, exist_ok=True)

        self.dstore_keys = np.memmap(f"{self.keys_vals_prefix}_keys.npy",
                        dtype=np.float16, mode="w+",
                        shape=(self.dstore_size, self.dimension))
        self.dstore_vals = np.memmap(f"{self.keys_vals_prefix}_vals.npy",
                        dtype=np.int32, mode="w+",
                        shape=(self.dstore_size, 1))

        dir_name = self.temp_save_dir_prefix[:self.temp_save_dir_prefix.rfind("/")]
        for file in os.listdir(dir_name):
            # file = f"{self.temp_save_dir_prefix}_temp_keys_{cur_size}_{batch_time_size}.npy"
            if "_temp_keys_" not in file:
                continue
            cur_size = int(file[file.find("_keys_") + len("_keys_"):file.rfind("_")])
            batch_size = int(file[file.rfind("_") + 1:file.find(".npy")])

            keys = np.memmap(f"{dir_name}/{file}", dtype=np.float16, mode='r',
                                shape=(batch_size, self.dimension))
            vals = np.memmap(f"{dir_name}/{file.replace('keys', 'vals')}", dtype=np.int32, mode='r',
                                shape=(batch_size, 1))
            
            self.dstore_keys[cur_size:cur_size + batch_size] = keys
            self.dstore_vals[cur_size:cur_size + batch_size] = vals
            os.remove(f"{dir_name}/{file.replace('keys', 'vals')}")
            os.remove(f"{dir_name}/{file}")

        self.combined_dataset = True
        logger.info('Combined dataset')

    def write_index(self):
        index_name = get_index_path(self.dstore_dir, self.model.config.model_type,
                                        self.cur_dstore_size, self.dimension) 
        Path(index_name).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f'Writing Index to {index_name}')
        start = time.time()
        faiss.write_index(self.index, f'{index_name}')
        logger.info(f'Writing index took {time.time() - start} s')

        logger.info('Moving vals...')
        self.dstore_size = self.cur_dstore_size
        self.keys_vals_prefix = get_dstore_path(self.dstore_dir, self.model.config.model_type,
                                                self.dstore_size, self.dimension)
        dstore_vals = np.memmap(f"{self.keys_vals_prefix}_vals.npy",
                        dtype=np.int32, mode="w+",
                        shape=(self.dstore_size, 1))
        vals_to_copy = np.memmap(f"{self.temp_save_dir_prefix}_temp_vals_{self.cur_dstore_size}.npy",
                        dtype=np.int32, mode="r",
                        shape=(self.cur_dstore_size, 1))
        dstore_vals[:,:] = vals_to_copy

        del vals_to_copy
        os.remove(f"{self.temp_save_dir_prefix}_temp_vals_{self.cur_dstore_size}.npy")

        logger.info('Moved vals')

    def update_index(self):
        dstore_keys = np.memmap(f"{self.temp_save_dir_prefix}_temp_keys.npy",
                        dtype=np.float16, mode="w+",
                        shape=(self.cur_dstore_size - self.total_keys_added, self.dimension))
        dstore_vals = np.memmap(f"{self.temp_save_dir_prefix}_temp_vals_{self.cur_dstore_size}.npy",
                        dtype=np.int32, mode="w+",
                        shape=(self.cur_dstore_size, 1))
        
        if self.total_keys_added > 0:
            combined_vals = f"{self.temp_save_dir_prefix}_temp_vals_{self.total_keys_added}.npy"
            vals = np.memmap(combined_vals, dtype=np.int32, mode='r',
                                    shape=(self.total_keys_added, 1))
            dstore_vals[:self.total_keys_added] = vals
            del vals
            os.remove(combined_vals)
        

        dir_name = self.temp_save_dir_prefix[:self.temp_save_dir_prefix.rfind("/")]
        cur_size = 0
        for file in os.listdir(dir_name):
            # file = f"{self.temp_save_dir_prefix}_temp_keys_{cur_size}_{batch_time_size}.npy"
            if "_temp_keys_" not in file:
                continue
            batch_size = int(file[file.rfind("_") + 1:file.find(".npy")])

            keys = np.memmap(f"{dir_name}/{file}", dtype=np.float16, mode='r',
                                  shape=(batch_size, self.dimension))
            vals = np.memmap(f"{dir_name}/{file.replace('keys', 'vals')}", dtype=np.int32, mode='r',
                                  shape=(batch_size, 1))
            
            dstore_keys[cur_size:cur_size + batch_size] = keys
            dstore_vals[self.total_keys_added + cur_size:self.total_keys_added + cur_size + batch_size] \
                = vals
            os.remove(f"{dir_name}/{file.replace('keys', 'vals')}")
            os.remove(f"{dir_name}/{file}")
            cur_size += batch_size

        self.total_keys_added += cur_size

        if not self.index.is_trained:
            logger.info('Training Index')
            np.random.seed(42)
            start = time.time()
            # Faiss does not handle adding keys in fp16 as of writing this.
            self.index.train(torch.tensor(dstore_keys.astype(np.float32)))
            logger.info(f'Training took {time.time() - start} s')

        logger.info('Adding Keys')
        start_time = time.time()
        self.index.add_with_ids(torch.tensor(dstore_keys.astype(np.float32)), torch.arange(dstore_keys.shape[0]))
        logger.info(f'Adding took {time.time() - start_time} s')

        del dstore_vals
        del dstore_keys
        os.remove(f"{self.temp_save_dir_prefix}_keys.npy")

    def register_hook(self, layer, func, pre=False):
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)
    
    def break_out(self):
        if self.build_dstore and not self.combined_dataset:
            self.combine_dataset()
        if self.build_index_on_the_go:
            if self.total_keys_added < self.cur_dstore_size:
                self.update_index()
            self.write_index()
        for h in self.hook_handles:
            h.remove()
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None

    def build_index(self, num_keys_to_add_at_a_time=1000000, 
            ncentroids=4096, seed=1, code_size=64, probe=32):
        if self.build_dstore and not self.combined_dataset:
            self.combine_dataset()
            
        logger.info('Building index')
        index_name = get_index_path(self.dstore_dir, self.model.config.model_type, self.dstore_size, self.dimension) 
        
        # Initialize faiss index
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFPQ(quantizer, self.dimension,
            ncentroids, code_size, 8)
        index.nprobe = probe

        logger.info('Training Index')
        np.random.seed(seed)
        random_sample = np.random.choice(np.arange(self.dstore_vals.shape[0]), size=[min(1000000, self.dstore_vals.shape[0])], replace=False)
        start = time.time()
        # Faiss does not handle adding keys in fp16 as of writing this.
        index.train(self.dstore_keys[random_sample].astype(np.float32))
        logger.info(f'Training took {time.time() - start} s')

        logger.info('Adding Keys')
        # index = faiss.read_index(f'{index_name}.trained')
        start = 0
        start_time = time.time()
        while start < self.dstore_size:
            end = min(self.dstore_size, start + num_keys_to_add_at_a_time)
            to_add = self.dstore_keys[start:end].copy()
            index.add_with_ids(torch.tensor(to_add.astype(np.float32)), torch.arange(start, end))
            start += num_keys_to_add_at_a_time

            if (start % 1000000) == 0:
                logger.info(f'Added {start} tokens so far')
                logger.info(f'Writing Index {start}')
                faiss.write_index(index, f'{index_name}')

        logger.info(f'Adding total {start} keys')
        logger.info(f'Adding took {time.time() - start_time} s')
        logger.info(f'Writing Index to {index_name}')
        start = time.time()
        faiss.write_index(index, f'{index_name}')
        logger.info(f'Writing index took {time.time() - start} s')
        
    def get_metrics(self):
        return {}

class ActivationCapturer(nn.Module):
    def __init__(self, layer, capture_input=False):
        super().__init__()
        self.layer = layer
        self.capture_input = capture_input

        self.captured = None

    
    def forward(self, module, input, output):
        if self.capture_input:
            self.captured = input[0].detach()
        else:
            self.captured = output.detach()


def get_dstore_path(dstore_dir, model_type, dstore_size, dimension):
    return f'{dstore_dir}/dstore_{model_type}_{dstore_size}_{dimension}'

def get_temp_dstore_path(dstore_dir, model_type, dimension):
    return f'{dstore_dir}/dstore_{model_type}_{dimension}'

def get_index_path(dstore_dir, model_type, dstore_size, dimension):
    return f'{dstore_dir}/index_{model_type}_{dstore_size}_{dimension}.indexed'
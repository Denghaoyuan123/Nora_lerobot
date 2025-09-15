
import torch
from torch import nn
import copy



def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim

import torch.nn.functional as F  # noqa: N812
import torch
from typing import Optional,Callable,Dict,Any
from torch import nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention,apply_multimodal_rotary_pos_emb,eager_attention_forward,repeat_kv
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
from transformers import Qwen2_5_VLTextModel,Qwen2_5_VLForConditionalGeneration
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers import AutoProcessor
from einops import rearrange, repeat
from qwen_vl_utils import process_vision_info
import PIL
import json
import math
import numpy as np
from huggingface_hub import hf_hub_download

def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
):
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = torch.float32
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb

def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)

def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks

class Qwen2_5_VLMoTAttention(Qwen2_5_VLAttention):
    """
   
    """

    def __init__(self, config: Qwen2_5_VLTextConfig, layer_idx: Optional[int] = None):
        super().__init__(config,layer_idx)
        

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        fill_kv_cache=True,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        
        #cos, sin = position_embeddings

        ## Since our action chunk is 1d time series, we do not need multimodal rope. Switch to normal rope instead
        #query_states, key_states = apply_multimodal_rotary_pos_emb(
        #    query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        #)
        query_states = rearrange(query_states, 'b h s d -> b s h d')
        query_states = apply_rope(query_states,position_ids)
        query_states = rearrange(query_states, 'b s h d -> b h s d')

        key_states = rearrange(key_states, 'b h s d -> b s h d')
        key_states = apply_rope(key_states,position_ids)
        key_states = rearrange(key_states, 'b s h d -> b h s d')
        
        
        if use_cache:
            
                past_key_state = past_key_value[self.layer_idx][0]
                past_value_state = past_key_value[self.layer_idx][1]
                
                key_states = torch.cat([past_key_state, key_states], dim=2)
               # print(key_states.dtype)
                value_states = torch.cat(
                    [past_value_state, value_states], dim=2
                )
                key_states = key_states.to(dtype=query_states.dtype)
                value_states = value_states.to(dtype=query_states.dtype)
                #print("New K shape",key_states.shape)
                #print("New V shape",value_states.shape)
        
        
        
        #if past_key_value is not None and not fill_kv_cache: ## Only update KV cache if fill_kv_cache is False
            #cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
           # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        #print("New query shape",query_states.shape)
        
        
        #attention_mask = torch.ones()
        ## I need to check if is_casual is default to True here. Is casual will automatically create an attention mask and I do not want that to happen.
        #print(position_ids)
        #print(attention_mask.shape)
       
        attn_output, attn_weights = attention_interface(
            self,
            query_states,  
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            position_ids=position_ids,  # pass positions for FA2
            **kwargs,
        )
        
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
from transformers.modeling_outputs import BaseModelOutputWithPast
class Qwen2_5_VLAExpert(Qwen2_5_VLTextModel):



    def __init__(self,config):
        super().__init__(config)

        

    def forward(self,
        expert_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        vlm_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[FlashAttentionKwargs],):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            raise ValueError("You must specify exactly inputs_embeds")
        # torch.jit.trace() doesn't support cache objects in the output
        if  vlm_key_values is None:
            raise ValueError("You must specify vlm_cache")

       
        

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        #position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=expert_attention_mask,
                position_ids=position_ids,
                past_key_value=vlm_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=None,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, vlm_key_values, all_hidden_states, all_self_attns] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=vlm_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    





class VLAWithExpert(nn.Module):
    def __init__(self,config=None,device=None):
        super().__init__()
        
        
        self.vlm  = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "declare-lab/nora-long",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        if config is not None:
            self.config = config
        else:
            self.config = {'max_action_dim':7,"max_state_dim":8}
        
        
        print("Loading expert model...")
        
        self.lm_expert_config = copy.deepcopy(self.vlm.config.text_config)

        #lm_expert_config = copy.deepcopy(model.config.text_config)
        self.processor = AutoProcessor.from_pretrained(
                "declare-lab/nora", trust_remote_code=True
            )
        self.fast_tokenizer = fast_tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )

        hidden_size = self.lm_expert_config.hidden_size
        expert_width_multiplier = 0.375
        self.lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)  # hidden_size // 2
        self.lm_expert_config.intermediate_size = get_intermediate_size(int(hidden_size * expert_width_multiplier))
        self.lm_expert_config.num_hidden_layers = self.vlm.config.num_hidden_layers
        self.lm_expert_config.num_attention_heads = 6

        self.action_expert = Qwen2_5_VLAExpert._from_config(self.lm_expert_config,torch_dtype=torch.bfloat16)
        self.action_chunk_length = 5
            
        self.device = self.vlm.device
        # Replace the action expert's attention layers
        
        self._replace_action_expert_attention()
        self.action_expert.embed_tokens = None
        self.vlm_kv_cache = None


       # self.state_proj = nn.Linear(
       #     self.config['max_state_dim'], hidden_size
       # )
        self.action_in_proj = nn.Linear(self.config['max_action_dim'],self.lm_expert_config.hidden_size)
        self.action_out_proj = nn.Linear(self.lm_expert_config.hidden_size, self.config['max_action_dim'])
        self.action_time_mlp_in = nn.Linear(
            self.lm_expert_config.hidden_size * 2, self.lm_expert_config.hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.lm_expert_config.hidden_size, self.lm_expert_config.hidden_size
        )
        
        self.device = self.vlm.device
        print(f"*** Loading normalization stats from HF Hub ***")
        norm_stats_path = hf_hub_download(repo_id='declare-lab/nora', filename="norm_stats.json")
        with open(norm_stats_path, "r") as f:
            self.norm_stats = json.load(f)

        libero_stats  = hf_hub_download(repo_id='moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10', filename="dataset_statistics.json")
        with open(libero_stats, "r") as f:
            self.norm_stats.update(json.load(f))
            

       
      
        
       
  

    def sample_noise(self, shape, device,dtype=torch.float32):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=dtype,
            device=device,
        )
        return noise
    def sample_time(self, bsize, device,dtype=torch.float32):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=dtype)
        time = time_beta * 0.999 + 0.001
        return time

    def _replace_action_expert_attention(self):
        """
        Iterate through the student model's layers and replace the default
        Qwen2_5_VLAttention with our custom Qwen2_5_VLMoTAttention.
        """
        for i, layer in enumerate(self.action_expert.layers):
            layer.self_attn = Qwen2_5_VLMoTAttention(
                config=self.action_expert.config, 
                layer_idx=i
            ).to(self.action_expert.dtype)
            layer.self_attn.to(self.action_expert.device)

    def precompute_vlm_kv_cache(self, vlm_inputs):
        """
        Run a forward pass on the expert model to generate and store its KV cache.
        """
        print("Pre-computing vlm KV cache...")
       
        vlm_outputs = self.vlm(
                **vlm_inputs,
                use_cache=True
            )
        self.vlm_kv_cache = vlm_outputs.past_key_values
        print("Vlm KV cache computed.")
    

    def denoise_step(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        vlm_kv_cache: tuple,
        full_2d_attn_mask: torch.Tensor):
        """
        Applies one denoising step to the noisy action `x_t` at a given `timestep`,
        conditioned on the VLM's output cache.

        This function is derived from the main `forward` pass, encapsulating the
        logic for a single step in the diffusion sampling process.

        Args:
            self: The instance of the model class.
            x_t (torch.Tensor): The noisy action tensor from the previous step.
                                Shape: (batch_size, action_chunk_length, action_dim).
            timestep (torch.Tensor): The current timestep for each sample in the batch.
                                    Shape: (batch_size,).
            vlm_kv_cache (tuple): The pre-computed key-value cache from the VLM,
                                used as conditioning.
            vlm_pad_mask (torch.Tensor): The padding mask for the VLM inputs, required
                                        to build the cross-attention mask.
                                        Shape: (batch_size, vlm_seq_len).

        Returns:
            torch.Tensor: The predicted noise `u_t` (epsilon).
                        Shape: (batch_size, action_chunk_length, action_dim).
        """
        device = x_t.device
        bsz = x_t.shape[0]

        # 1. Embed the noisy action `x_t`
        x_t = x_t.to(dtype=self.vlm.dtype)

        action_input_embeds = self.action_in_proj(x_t)

        # 2. Create sinusoidal time embeddings from the current timestep
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.lm_expert_config.hidden_size,
            4e-3, # Values from your forward pass
            4.0,
            device=device,
        )
        time_emb = time_emb.type(dtype=x_t.dtype)
        # Expand time embedding to match the action embedding dimensions
        time_emb = time_emb[:, None, :].expand_as(action_input_embeds)

        # 3. Combine action and time embeddings and process through MLPs
        action_time_emb = torch.cat([action_input_embeds, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish activation
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # 4. Construct the attention mask for the action expert.
        # The expert needs to attend to the VLM context and its own action inputs.
        
        
        # The expert's queries originate from the action sequence, so we slice the mask accordingly.
        # It can attend to the full VLM context and the action sequence.
        expert_attention_mask = full_2d_attn_mask[:, -self.action_chunk_length:, :]

        # 5. Prepare position_ids for the expert.
        # Note: This implementation mirrors your forward pass, where position_ids for the
        # expert restart from 0.
        position_ids = torch.arange(self.action_chunk_length, device=device)

        # 6. Call the action expert with the prepared inputs and VLM cache.
        expert_output = self.action_expert(
            inputs_embeds=action_time_emb,
            expert_attention_mask=expert_attention_mask.unsqueeze(1).bool(), # Add head dim
            position_ids=position_ids,
            vlm_key_values=vlm_kv_cache,
            use_cache=True, # As in the original forward pass
        )

        # 7. Project the expert's output to get the final noise prediction.
        velocity = self.action_out_proj(expert_output.last_hidden_state)

        return velocity

   
    @torch.no_grad()
    def sample_actions(self, image,instruction: dict,num_steps:int = 25,unnorm_key='libero_10',unnormalize=True):
        """
        Generates actions by running the full diffusion sampling process.

        This function first computes the VLM's key-value cache to use as a
        conditioning context. It then uses an iterative Euler-method-based
        sampler, calling `denoise_step` at each timestep to refine a noise
        tensor into a final action.

        Args:
            self: The instance of the model class.
            vlm_inputs (dict): A dictionary containing the inputs for the VLM,
                            e.g., {'input_ids': ..., 'attention_mask': ...}.
            noise (Tensor, optional): An initial noise tensor to start the
                                    sampling from. If None, it will be
                                    sampled randomly. Defaults to None.
                                    Shape: (batch_size, action_chunk_length, action_dim).

        Returns:
            Tensor: The final, denoised action tensor.
                    Shape: (batch_size, action_chunk_length, action_dim).
        """
        #vlm_inputs = self.prepare_inputs_for_generation(image,instruction)
        device = self.vlm.device
        #print(type(image))
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)
                # Construct messages in the expected chat format. Note that nora expects image of size 224 by 224
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "resized_height": 224,
                        "resized_width": 224,
                    },
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        # Apply chat template to get the text input for the model
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

                # Process vision information (depends on your process_vision_info function)
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs for the model using the main processor
        #image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to GPU
       
        inputs = {k: v.to(device) for k, v in inputs.items()}

         
    
       
        bsz = inputs['input_ids'].shape[0]
       

        

# 1. Pre-compute the VLM cache. This context is the conditioning for the
#    entire denoising process and only needs to be computed once.

        vlm_outputs = self.vlm(**inputs)
        vlm_kv_cache = vlm_outputs.past_key_values
        # The VLM's attention mask is its padding mask for the expert.
        vlm_pad_mask = inputs['attention_mask'].clone()

    # 2. Initialize the noisy action tensor `x_t`.

        actions_shape = (bsz, self.action_chunk_length, self.config['max_action_dim'])
        x_t = self.sample_noise(actions_shape, device=device,dtype=self.vlm.dtype)


        # 3. Set up the time steps for the Euler solver.
        # We will step from t=1 down to t=0.
        #num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        dt_tensor = torch.tensor(dt, dtype=self.vlm.dtype, device=device)
        time = torch.tensor(1.0, dtype=self.vlm.dtype, device=device)

        # 4. Iteratively denoise using the Euler method.
        # The loop continues as long as time is greater than or equal to zero.
        action_pad_mask = torch.ones(bsz, self.action_chunk_length, device=device).bool()
        
        # An all-zero attention mask for the action part allows for full bidirectional attention
        # within the action chunk, as seen in the original forward pass.
        action_attn_mask = torch.zeros(bsz, self.action_chunk_length, device=device).bool()

        # Concatenate VLM (prefix) and action masks.
        # The VLM's attention mask is its padding mask.
        concat_pad_mask = torch.cat([vlm_pad_mask, action_pad_mask], dim=1)
        concat_attn_mask = torch.cat([vlm_pad_mask, action_attn_mask], dim=1)

        # Create the full 2D attention mask for the combined sequence.
        full_2d_attn_mask = make_att_2d_masks(concat_pad_mask, concat_attn_mask)
        while time >= -dt / 2: # Loop until t=0
            with torch.no_grad():
                # Expand the current time to match the batch size.
                expanded_time = time.expand(bsz)

                # Call the denoise_step function to predict the velocity v_t (or noise u_t).
                # The function takes the current noisy action, timestep, and the
                # pre-computed VLM cache and padding mask as input.
                #print(expanded_time)
                v_t = self.denoise_step(
                    x_t=x_t,
                    timestep=expanded_time,
                    vlm_kv_cache=vlm_kv_cache,
                    full_2d_attn_mask=full_2d_attn_mask,
                )

                # 5. Apply the Euler integration step to update the action tensor.
                # This moves the action slightly along the direction of the predicted velocity.
                x_t += dt * v_t
                time += dt

        # 6. Return the final denoised action.
        normalized_action = x_t.cpu().float().numpy()
        if not unnormalize:
            
            return normalized_action
        action_stats = self._get_action_stats(unnorm_key)

        mask = action_stats.get("mask", np.ones_like(action_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_stats["q99"]), np.array(action_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_action + 1) * (action_high - action_low) + action_low,
            normalized_action,
        )
        return actions
    
    def _get_action_stats(self, unnorm_key: str) -> Dict[str, Any]:
        if unnorm_key not in self.norm_stats:
            raise KeyError(
                f"The `unnorm_key` '{unnorm_key}' is not in the set of available dataset statistics. "
                f"Please choose from: {list(self.norm_stats.keys())}"
            )
        return self.norm_stats[unnorm_key]["action"]
    def forward(self,vlm_inputs, actions,alpha=10.0, **kwargs):
        """
        The main forward pass that uses the student model with the expert's cache.
        """
        
            
        # The magic happens here: we pass the expert cache into the student's forward call.
        # This will require modifying how arguments are passed down.
        ## Precompute the VLM cache with only VLM inputs/attention mask 
        ## Let the Qwen2_5 vlm settle its own attention mask. 
        device = self.vlm.device
        vlm_outputs = self.vlm(
                **vlm_inputs,
                use_cache=True
            )
        self.vlm_kv_cache = vlm_outputs.past_key_values

        ## Construct attention mask for the action expert.
        ## The action expert should be able to attend to the VLM inputs and its own action inputs. ( Prefix + bidirectional attention)

        bsz = vlm_inputs['input_ids'].shape[0]
        vlm_pad_mask = vlm_inputs['expert_attention'].clone()
        vlm_attn_mask = vlm_inputs['attention_mask'].clone()

        
        
        actions = actions.to(self.vlm.dtype)
        noise = self.sample_noise(actions.shape, actions.device,dtype=actions.dtype)

        
        time = self.sample_time(actions.shape[0], actions.device,dtype=actions.dtype)
        
        

        time_expanded = time[:, None, None]
        

        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        #x_t = x_t.to(self.vlm.dtype)
        action_input_embeds = self.action_in_proj(x_t) ## Embed noisy action
        
        time_emb = create_sinusoidal_pos_embedding(
            time,
            self.lm_expert_config.hidden_size,
            4e-3,
            4.0,
            device=device,
        )

        time_emb = time_emb.type(dtype=actions.dtype)

        time_emb = time_emb[:, None, :].expand_as(action_input_embeds)

        
        action_time_emb = torch.cat([action_input_embeds, time_emb], dim=2) ## concat on the hidden size dim

        action_time_emb = self.action_time_mlp_in(action_time_emb) ## simple linear layer to project back to hidden size dim
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb) ## 


        


    
        action_pad_mask = torch.ones(bsz,self.action_chunk_length,device=device).bool()
        action_attn_mask = torch.zeros(bsz,self.action_chunk_length,device=device).bool()

        concat_action_mask = torch.cat([vlm_pad_mask,action_pad_mask],dim=1)
        concat_attn_mask = torch.cat([vlm_attn_mask,action_attn_mask],dim=1)

        attn = make_att_2d_masks(concat_action_mask,concat_attn_mask)
        expert_attention_mask = attn[:, -self.action_chunk_length:, : vlm_pad_mask.shape[1]+self.action_chunk_length :]
        
        
        position_ids = torch.arange(self.action_chunk_length,device=device)
        expert_output = self.action_expert(inputs_embeds=action_time_emb,
                                    expert_attention_mask=expert_attention_mask.unsqueeze(1).bool(),
                                    position_ids= position_ids,
                                    vlm_key_values=self.vlm_kv_cache, 
                                    use_cache=True)
        action_out = self.action_out_proj(expert_output.last_hidden_state)
        expert_loss = alpha*F.mse_loss(action_out, u_t, reduction='mean')
        
        loss = expert_loss+ vlm_outputs.loss
        
        return {'expert_loss': expert_loss,'combined_loss':loss,'vlm_loss':vlm_outputs.loss}

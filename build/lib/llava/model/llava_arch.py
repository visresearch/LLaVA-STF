#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector, build_vision_projector_baseline
from .multimodal_projector.builder import build_vision_projector_crosslayer, build_vision_projector_neighborlayer, build_local_attention

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        # self.pos_2d = self.build_2d_sincos_position_embedding((24,24), 1024)
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
            # self.mm_projector_crosslayer = build_vision_projector_crosslayer(config)
            # self.mm_projector_neighborlayer1 = build_vision_projector_neighborlayer(config)
            # self.mm_projector_neighborlayer2 = build_vision_projector_neighborlayer(config)
            # self.mm_projector_neighborlayer3 = build_vision_projector_neighborlayer(config)
            # self.mm_projector_neighborlayer4 = build_vision_projector_neighborlayer(config)
            # self.mm_projector_neighborlayer5 = build_vision_projector_neighborlayer(config)
            # self.mm_projector_neighborlayer6 = build_vision_projector_neighborlayer(config)
            # self.mm_projector_neighborlayer7 = build_vision_projector_neighborlayer(config)
            # self.mm_projector_neighborlayer8 = build_vision_projector_neighborlayer(config)
            # self.linear_projector = build_linear_projector(config)
            # self.local_attn = build_local_attention()
            # self.mm_projector_teacher = build_vision_projector_teacher(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_linear_projector(self):
        linear_proj = getattr(self, 'linear_projector', None)
        return linear_proj
    
    def get_local_attn(self):
        local_attn = getattr(self, 'local_attn', None)
        return local_attn
    
    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        # pretrain_mm_mlp_adapter_teacher = model_args.pretrain_mm_mlp_adapter_teacher
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        # if self.get_linear_projector() is None:
        #     self.linear_projector = build_linear_projector(self.config)
        # if self.get_local_attn() is None:
        #     self.local_attn = build_local_attention()

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
            # self.mm_projector_crosslayer = build_vision_projector_crosslayer(self.config)
            # self.mm_projector_neighborlayer1 = build_vision_projector_neighborlayer(self.config)
            # self.mm_projector_neighborlayer2 = build_vision_projector_neighborlayer(self.config)
            # self.mm_projector_neighborlayer3 = build_vision_projector_neighborlayer(self.config)
            # self.mm_projector_neighborlayer4 = build_vision_projector_neighborlayer(self.config)
            # self.mm_projector_neighborlayer5 = build_vision_projector_neighborlayer(self.config)
            # self.mm_projector_neighborlayer6 = build_vision_projector_neighborlayer(self.config)
            # self.mm_projector_neighborlayer7 = build_vision_projector_neighborlayer(self.config)
            # self.mm_projector_neighborlayer8 = build_vision_projector_neighborlayer(self.config)
            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True
            # for p in self.mm_projector_crosslayer.parameters():
            #     p.requires_grad = True
            # for p in self.mm_projector_neighborlayer1.parameters():
            #     p.requires_grad = True
            # for p in self.mm_projector_neighborlayer2.parameters():
            #     p.requires_grad = True
            # for p in self.mm_projector_neighborlayer3.parameters():
            #     p.requires_grad = True
            # for p in self.mm_projector_neighborlayer4.parameters():
            #     p.requires_grad = True
            # for p in self.mm_projector_neighborlayer5.parameters():
            #     p.requires_grad = True
            # for p in self.mm_projector_neighborlayer6.parameters():
            #     p.requires_grad = True
            # for p in self.mm_projector_neighborlayer7.parameters():
            #     p.requires_grad = True
            # for p in self.mm_projector_neighborlayer8.parameters():
            #     p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            print("------------------------------------- load pretrained mlp --------------------------------------")
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            print(f'projector keys: {mm_projector_weights.keys()}')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            
            # ------------------------------------- baseline --------------------------------------
            # missing_keys, unexpected_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            # print('missing_keys student mm_projector:', missing_keys)
            # print('unexpected_keys student mm_projector:', unexpected_keys)

            # 
            missing_keys, unexpected_keys = self.load_state_dict(get_w(mm_projector_weights, 'model'), strict=False)
            # print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def split_image_features(self, image_features, window_size, concat=True):
        b, n, c = image_features.shape
        h = w = int(n ** 0.5) 
        ws = window_size
        dim_ratio = ws * ws
        image_features = image_features.view(b, h, w, c)
        image_features = image_features.unfold(1, ws, ws).unfold(2, ws, ws) 
        if concat:
            image_features = image_features.contiguous().view(b, h//ws, w//ws, dim_ratio*c)
            image_features = image_features.view(b, -1, dim_ratio*c)
        else:
            image_features = image_features.contiguous().view(b, h//ws, w//ws, ws*ws, c)
            image_features = image_features.view(b, -1, ws*ws, c)
        return image_features
    
    def encode_images_distill(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        with torch.no_grad():
            image_features_teacher = self.get_model().mm_projector_teacher(image_features)
        image_features_student = self.split_image_features(image_features, 2)
        image_features_student = self.get_model().mm_projector(image_features_student)
        return image_features_student, image_features_teacher
    
    def encode_images(self, images):
        # --------------------------------------------- Single 2x2 Merge ---------------------------------------------
        # image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.split_image_features(image_features, 2)
        # image_features = self.get_model().mm_projector(image_features) 
        # ------------------------------------------------------------------------------------------------------------
        
        # --------------------------------------------- Mix 2x2 Merge ------------------------------------------------
        # image_features = self.get_model().get_vision_tower()(images)
        # b, n, c = image_features.shape
        # h = w = int(n ** 0.5)
        # image_features = image_features.reshape(b, h // 4, 4, w // 4, 4, c)
        # image_features = image_features.permute(0, 1, 3, 2, 4, 5)
        # b, h, w, ws, ws, c = image_features.shape
        # image_features = image_features.contiguous().view(b, h, w, -1, c)
        # order = [0, 3, 2, 9, 12, 15, 14, 5, 8, 11, 10, 1, 4, 7, 6, 13]
        # image_features = image_features[:, :, :, order, :]
        # image_features = image_features.contiguous().view(b, -1, 4*c)
        # image_features = self.get_model().mm_projector(image_features)
        # ------------------------------------------------------------------------------------------------------------
        
        # --------------------------------------------- Later Merge ------------------------------------------------
        # image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().mm_projector(image_features)
        # image_features = self.split_image_features(image_features, 2)
        # image_features = self.get_model().linear_projector(image_features)

        # --------------------------------------------- Local Attn ------------------------------------
        # image_features = self.get_model().get_vision_tower()(images)
        # # image_features = image_features + self.get_model().pos_2d
        # image_features = self.split_image_features(image_features, 2, False)
        # image_features = self.get_model().local_attn(image_features)
        # image_features = self.get_model().mm_projector(image_features)
        # print(f'image shape: {image_features.shape}')
        # ---------------------------------------------------------------------------------------------

        # --------------------------------------------- Cross Layer ------------------------------------
        # image_features, image_features_l21, image_features_l18, image_features_l15, image_features_l12, image_features_l9, image_features_l6, image_features_l3 = self.get_model().get_vision_tower()(images)
        # image_features = torch.cat((image_features,image_features_l21,image_features_l18,image_features_l15,image_features_l12,image_features_l9,image_features_l6,image_features_l3), dim=-1) #image_features_l8,image_features_l12,
        # # image_features = torch.cat((image_features,image_features_l21), dim=-1)

        # image_features = self.get_model().get_vision_tower()(images)
        # # -----branch 1------
        # image_features_layers = (image_features[-2],)#image_features[-2],
        # for i in range(8):
        #     image_features_layer = torch.cat((image_features[3*i],image_features[3*i+1],image_features[3*i+2]),dim=-1)
        #     if i == 0:
        #         # image_features_layer = self.get_model().mm_projector_neighborlayer1(image_features_layer)
        #         image_features_layer = image_features[2]
        #     elif i == 1:
        #         # image_features_layer = self.get_model().mm_projector_neighborlayer2(image_features_layer)
        #         image_features_layer = image_features[5]
        #     elif i == 2:
        #         # image_features_layer = self.get_model().mm_projector_neighborlayer3(image_features_layer)
        #         image_features_layer = image_features[8]
        #     elif i == 3:
        #         image_features_layer = self.get_model().mm_projector_neighborlayer4(image_features_layer)
        #     elif i == 4:
        #         image_features_layer = self.get_model().mm_projector_neighborlayer5(image_features_layer)
        #     elif i == 5:
        #         image_features_layer = self.get_model().mm_projector_neighborlayer6(image_features_layer)
        #     elif i == 6:
        #         image_features_layer = self.get_model().mm_projector_neighborlayer7(image_features_layer)
        #     elif i == 7:
        #         image_features_layer = self.get_model().mm_projector_neighborlayer8(image_features_layer)
        #     image_features_layers = image_features_layers + (image_features_layer,)
        # image_features = torch.cat(image_features_layers,dim=-1)

        # -----branch 2-----
        # image_features_layer1 = image_features[11]
        # image_features_layer2 = torch.cat((image_features[12],image_features[13],image_features[14],image_features[15]),dim=-1)
        # image_features_layer3 = torch.cat((image_features[16],image_features[17],image_features[18],image_features[19]),dim=-1)
        # image_features_layer4 = torch.cat((image_features[20],image_features[21],image_features[22],image_features[23]),dim=-1)
        # image_features_layer2 = self.get_model().mm_projector_neighborlayer2(image_features_layer2)
        # image_features_layer3 = self.get_model().mm_projector_neighborlayer3(image_features_layer3)
        # image_features_layer4 = self.get_model().mm_projector_neighborlayer4(image_features_layer4)
        # image_features = torch.cat((image_features_layer1,image_features_layer2,image_features_layer3,image_features_layer4),dim=-1)
        
        # -----branch 3-----
        # image_features_layers = (image_features[-2],)
        # for i in range(8):
        #     image_features_layer = (image_features[3*i]+image_features[3*i+1]+image_features[3*i+2])/3
        #     # image_features_layer = 
        #     image_features_layers = image_features_layers + (image_features_layer,)
        # image_features = torch.cat(image_features_layers,dim=-1)

        # -----branch 4-----
        # image_features_1 = []
        # image_features_2 = []
        # image_features_3 = []
        # for i in range(0, 8):
        #     image_features_1.append(image_features[i])
        # image_features_1 = torch.stack(image_features_1, dim=0)
        # image_features_1 = torch.sum(image_features_1, dim=0) / 8
        # for i in range(8, 16):
        #     image_features_2.append(image_features[i])
        # image_features_2 = torch.stack(image_features_2, dim=0)
        # image_features_2 = torch.sum(image_features_2, dim=0) / 8
        # for i in range(16, 24):
        #     image_features_3.append(image_features[i])
        # image_features_3 = torch.stack(image_features_3, dim=0)
        # image_features_3 = torch.sum(image_features_3, dim=0) / 8
        # image_features = torch.cat([image_features[-2],image_features_1,image_features_2,image_features_3],dim=-1)

        # image_features = self.get_model().mm_projector_crosslayer(image_features)
        # image_features = self.split_image_features(image_features, 2)
        # image_features = self.get_model().mm_projector(image_features)
        # ----------------------------------------------------------------------------------------------

        # --------------------------------------------- Baseline -----------------------------------------------------
        image_features = self.get_model().get_vision_tower()(images)
        # print(f'image shape: {image_features.shape}')
        image_features = self.get_model().mm_projector(image_features)
        # ------------------------------------------------------------------------------------------------------------
        
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        img_pos = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                img_pos.append(2048)
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            img_pos.append(cur_input_embeds_no_im[0].shape[0])
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def prepare_inputs_labels_for_multimodal_distill(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        # print(f"input_ids: {input_ids}")
        # print(f"labels: {labels}")
        # print("--------------------------------------------------------")
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # print(type(images)) torch.Tensor
        # print(images.ndim) 4
        if type(images) is list or images.ndim == 5:
            print("tttttttttttttttttttttttttttttttttttttttttt images_type list list list list tttttttttttttttttttttttttttttttttttttttttttttttttttttttt")
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            # image_features = self.encode_images_teacher(concat_images)
            image_features, image_features_teacher = self.encode_images_distill(images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features_teacher = torch.split(image_features_teacher, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')

            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
                image_features_teacher = [x.flatten(0, 1) for x in image_features_teacher]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                new_image_features_teacher = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features, image_features_teacher = self.encode_images_distill(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        t_labels = labels
        t_position_ids = position_ids
        t_attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else: # this
            attention_mask = attention_mask.bool()
        if position_ids is None: # this
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        # print(f"input ids: {input_ids.shape}")
        # print(f"attention_mask: {attention_mask}")
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        # print(f'labels: {labels.shape}')
        # print(f'inputs ids {input_ids}') [tensor([长度])]
        new_input_embeds = []
        new_labels = []
        new_input_embeds_t = []
        new_labels_t = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                # print("000000000000000000000000000000000 images 000000000000000000000000000000000000000000000000000000000000")
                cur_image_features = image_features[cur_image_idx]
                cur_image_features_teacher = image_features_teacher[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                cur_input_embeds_teacher = torch.cat([cur_input_embeds_1, cur_image_features_teacher[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                new_input_embeds_t.append(cur_input_embeds_teacher)
                new_labels_t.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1): # [[...], [...]]：从-200位置，将id划分为两部分
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim] #[len1, len2]：划分后两部分各自的长度
            # print(f'split_sizes{split_sizes}')
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim)) #两部分拼接起来得到词嵌入
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0) #(tensor(...), tensor(...)):又将词嵌入分成两部分
            # print(f'cur_input_embeds_no_im {cur_input_embeds_no_im}')
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_input_embeds_t = []
            cur_new_labels_t = []
            
            for i in range(num_images + 1): #拼接图像特征
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_input_embeds_t.append(cur_input_embeds_no_im[i])
                cur_new_labels_t.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_features_t = image_features_teacher[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds_t.append(cur_image_features_t)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_new_labels_t.append(torch.full((cur_image_features_t.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

            cur_new_input_embeds_t = [x.to(self.device) for x in cur_new_input_embeds_t]
            cur_new_input_embeds_t = torch.cat(cur_new_input_embeds_t)
            cur_new_labels_t = torch.cat(cur_new_labels_t)
            new_input_embeds_t.append(cur_new_input_embeds_t)
            new_labels_t.append(cur_new_labels_t)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_input_embeds_t = [x[:tokenizer_model_max_length] for x in new_input_embeds_t]
            new_labels_t = [x[:tokenizer_model_max_length] for x in new_labels_t]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        max_len_t = max(x.shape[0] for x in new_input_embeds_t)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        new_input_embeds_padded_t = []
        new_labels_padded_t = torch.full((batch_size, max_len_t), IGNORE_INDEX, dtype=new_labels_t[0].dtype, device=new_labels_t[0].device)
        attention_mask_t = torch.zeros((batch_size, max_len_t), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids_t = torch.zeros((batch_size, max_len_t), dtype=position_ids.dtype, device=position_ids.device)
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds_t, new_labels_t)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded_t.append(torch.cat((
                    torch.zeros((max_len_t - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded_t[i, -cur_len:] = cur_new_labels
                    attention_mask_t[i, -cur_len:] = True
                    position_ids_t[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded_t.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len_t - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded_t[i, :cur_len] = cur_new_labels
                    attention_mask_t[i, :cur_len] = True
                    position_ids_t[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds_t = torch.stack(new_input_embeds_padded_t, dim=0)

        if _labels is None:
            new_labels = None
            new_labels_t = None
        else:
            new_labels = new_labels_padded
            new_labels_t = new_labels_padded_t


        if _attention_mask is None:
            attention_mask = None
            attention_mask_t = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)
            attention_mask_t = attention_mask_t.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
            position_ids_t = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, None, position_ids_t, attention_mask_t, past_key_values, new_input_embeds_t, new_labels_t
    
    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

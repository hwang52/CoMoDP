import torch
import torch.nn as nn
import math
from functools import reduce
from operator import mul
from models.procdf_diffusion import SDFeaturizer
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import VisionTransformer, PatchEmbed
from torch.nn.modules.utils import _pair 
import torch.nn.functional as F


class ProSDF_ViT(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, 
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None, Prompt_Token_num=1, 
                 VPT_type="Shallow", basic_state_dict=None, dift_input_dim=1280, new_classes=100, args=None):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         norm_layer=norm_layer, act_layer=act_layer)
        # load basic state_dict
        if basic_state_dict is not None:
            self.load_state_dict(basic_state_dict, False)
        # setting
        self.token_num = Prompt_Token_num
        self.dift_extractor = SDFeaturizer(args=args)

        self.meta_prompt_generator = nn.Sequential(
            nn.Linear(dift_input_dim, embed_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim//2, embed_dim)
        )
    

    def new_head(self, new_classes): # projection linear
        self.head = nn.Linear(self.embed_dim, new_classes) 


    def Freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.meta_prompt_generator.parameters():
            param.requires_grad = True
        for param in self.head.parameters():
            param.requires_grad = True


    def UnFreeze(self):
        for param in self.parameters():
            param.requires_grad = True


    def obtain_prompt(self):
        prompt_state_dict = {'head': self.head.state_dict(), 'meta': self.meta_prompt_generator.state_dict()}
        return prompt_state_dict


    def load_prompt(self, prompt_state_dict):
        try:
            self.head.load_state_dict(prompt_state_dict['head'], False)
        except:
            print('head not match, so skip head')
        else:
            print('prosdf head match')
        try:
            self.meta_prompt_generator.load_state_dict(prompt_state_dict['meta'], False)
        except:
            print('meta_prompt_generator not match, so skip head')
        else:
            print('prosdf meta_prompt_generator match')
    

    def AvgPool_fea(self, dift_fea):
        # dift_fea [b,1280,14,14]
        # reference: Refining activation downsampling with SoftPool https://zhuanlan.zhihu.com/p/344269280 
        e_dift = torch.sum(torch.exp(dift_fea), dim=1, keepdim=True) # b,1,14,14 
        # h_out = (h_in - kernel_size) / stride + 1, kernel_size, h_out = h_in / stride
        kernel_size, stride = _pair(dift_fea.size(-1)), _pair(dift_fea.size(-1))
        avg_fea = F.avg_pool2d(dift_fea.mul(e_dift), kernel_size=kernel_size, stride=stride)
        avg_e_fea = F.avg_pool2d(e_dift, kernel_size=kernel_size, stride=stride)
        dift_fea = avg_fea.mul_(sum(kernel_size)).div_(avg_e_fea.mul_(sum(kernel_size)))  # b,1280,1,1
        dift_fea = dift_fea.reshape(dift_fea.size(0), dift_fea.size(1)) # b,1280 
        return dift_fea


    def forward_features(self, x, prompt_tokens): # forward
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # concatenate CLS token
        x = torch.cat((cls_token, x), dim=1)
        # add position embedding
        x = self.pos_drop(x + self.pos_embed)

        Prompt_Token_num = prompt_tokens.shape[1]
        # concatenate Prompt_Tokens
        # Prompt_Tokens = self.Prompt_Tokens.expand(x.shape[0], -1, -1)
        Prompt_Tokens = prompt_tokens # b, num, emb_dim=768
        x = torch.cat((x, Prompt_Tokens), dim=1) 
        num_tokens = x.shape[1]
        # Sequntially procees
        x = self.blocks(x)[:, :num_tokens - Prompt_Token_num]
        x = self.norm(x)
        return x
    

    def diffusion_prompt_tokens(self, img, text_prompts):
        img = torch.tanh(img) # match -1,1 
        dift_fea = self.dift_extractor.forward(img_tensor=img, prompt=text_prompts, t=0, up_ft_index=1)
        # dift_fea = dift_fea.reshape(dift_fea.size(0), dift_fea.size(1), -1).mean(dim=-1) # b,1280 
        dift_fea = dift_fea.reshape(dift_fea.size(0), dift_fea.size(1), -1).norm(dim=-1) # b,1280
        
        list_tokens = [] 
        if self.token_num == 1:
            token = self.meta_prompt_generator(dift_fea) # [b, emb_dim]
            prompt_tokens = token.reshape(token.size(0), 1, token.size(1)) # [b, 1, emb_dim]
        else:
            for _ in range(self.token_num):
                token = self.meta_prompt_generator(dift_fea) # [b, emb_dim]
                list_tokens.append(token)
            prompt_tokens = torch.stack(list_tokens, dim=1) # [b, token_num, emb_dim] 
        
        # list_tokens = []
        # img = torch.tanh(img)
        # for _ in range(5):
        #     dift_fea = self.dift_extractor.forward(img_tensor=img, prompt=text_prompts, t=0, up_ft_index=1)
        #     dift_fea = dift_fea.detach()
        #     dift_fea = dift_fea.reshape(dift_fea.size(0), dift_fea.size(1), -1).mean(dim=-1)
        #     single_token = self.meta_prompt_generator(dift_fea)
        #     list_tokens.append(single_token)
        # prompt_tokens = torch.stack(list_tokens, dim=1) # [b, token_num, emb_dim]
        return prompt_tokens 
    

    def forward(self, x, text_prompts):
        prompt_tokens = self.diffusion_prompt_tokens(x, text_prompts)
        x = self.forward_features(x, prompt_tokens)
        cls_fea = x[:, 0, :]
        # use cls token for cls head
        x = self.fc_norm(x[:, 0, :]) # fixme for old timm: x = self.pre_logits(x[:, 0, :])
        x = self.head(x)
        return cls_fea, x


def build_prosdg_model(num_classes=100, img_size=224, model_idx='ViT', patch_size=16, 
                       Prompt_Token_num=5, VPT_type="Shallow", dift_input_dim=1280, args=None):
    # VPT_type = "Deep" / "Shallow"
    if model_idx[0:3] == 'ViT':
        # ViT_Prompt
        import timm
        from pprint import pprint
        model_names = timm.list_models('*vit*')
        # pprint(model_names)
        # get pretrained vit model
        basic_model = timm.create_model('vit_base_patch' + str(32) + '_' + str(img_size), pretrained=True)
        model = ProSDF_ViT(img_size=img_size, patch_size=patch_size, 
                           Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type, 
                           dift_input_dim=dift_input_dim, new_classes=num_classes, args=args)
        model.load_state_dict(basic_model.state_dict(), False)
        model.new_head(new_classes=num_classes)
        model.Freeze()
    else:
        print("The model is not difined in the Prompt script!") 
        return -1
    return model 
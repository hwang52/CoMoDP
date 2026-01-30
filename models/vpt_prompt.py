import timm
import torch
from .vpt_base_vit import *


def build_vpt_promptmodel(num_classes=10, img_size=224, model_idx='ViT', patch_size=16, 
                          Prompt_Token_num=8, VPT_type="Deep"):
    # VPT_type = "Deep" / "Shallow"
    if model_idx[0:3] == 'ViT':
        # ViT_Prompt
        import timm
        from pprint import pprint
        model_names = timm.list_models('*vit*')
        # pprint(model_names)
        # get pretrained vit model
        basic_model = timm.create_model('vit_base_patch' + str(32) + '_' + str(img_size), pretrained=True)
        model = VPT_ViT(img_size=img_size, patch_size=patch_size, Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type)
        model.load_state_dict(basic_model.state_dict(), False)
        model.New_CLS_head(num_classes)
        model.Freeze()
    else:
        print("The model is not difined in the Prompt script!")
        return -1
    return model

    # try:
    # img = torch.randn(10, 3, img_size, img_size)
    # preds = model(img)  # (1, class_number)
    # print('test model output shape: ', preds.shape)
    # return model
    # except:
    #     print("Problem exist in the model defining process!")
    #     return -1
    # else:
    #     print('model is ready now!')
    #     return model
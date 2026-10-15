import importlib.util
import os
import unittest
from cwru_bearing.data import load_config

HAS_TORCH = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('torchvision') is not None
if os.environ.get('CWRU_REQUIRE_TORCH') == '1' and not HAS_TORCH:
    raise RuntimeError('Model CI requires torch and torchvision; skipping is not allowed')


@unittest.skipUnless(HAS_TORCH,'torch/torchvision optional dependencies unavailable')
class ModelTests(unittest.TestCase):
    def test_cpu_forward_shape(self):
        import torch
        from cwru_bearing.models import build_model
        previous=torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            model=build_model(load_config('cross_fault_size'),pretrained=False).eval()
            with torch.no_grad():
                output=model(torch.zeros(1,3,224,224))
            self.assertEqual(tuple(output.shape),(1,4))
            self.assertTrue(torch.isfinite(output).all())
        finally:
            torch.set_num_threads(previous)

    def test_stage_parameters_and_modes(self):
        from torch import nn
        from cwru_bearing.models import build_model,configure_stage,set_training_mode
        for profile,first,second in [('cross_load',12810,424970),('cross_fault_size',5124,417284)]:
            model=build_model(load_config(profile),pretrained=False)
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),first)
            configure_stage(model,2)
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),second)
            set_training_mode(model,'features_eval')
            self.assertFalse(model.features.training)
            self.assertTrue(model.classifier.training)
            set_training_mode(model,'last_projection_train')
            self.assertFalse(model.features[0].training)
            self.assertTrue(model.features[-1].training)
            set_training_mode(model,'all_bn_eval')
            self.assertTrue(model.features[0].training)
            self.assertTrue(all(not m.training for m in model.modules() if isinstance(m,nn.BatchNorm2d)))

    def test_image_transform_mapping(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        import numpy as np
        from cwru_bearing.models import build_loaders
        with tempfile.TemporaryDirectory() as d:
            for split in ['train','val','test']:
                for c in range(4):
                    p=Path(d)/split/str(c);p.mkdir(parents=True)
                    Image.fromarray(((np.arange(257)[:,None]*7 + np.arange(57)[None,:]*13) % 256).astype(np.uint8)).save(p/'sample.png')
            loaders=build_loaders(d,load_config('cross_fault_size'))
            x,y=next(iter(loaders['test']))
            self.assertEqual(tuple(x.shape),(4,3,224,224))
            self.assertEqual(y.tolist(),[0,1,2,3])
            from torchvision import transforms
            with Image.open(Path(d)/'test/0/sample.png') as im:
                expected=transforms.Compose([transforms.Resize((224,224),antialias=True),
                    transforms.Grayscale(3),transforms.ToTensor(),transforms.Normalize(
                    [.485,.456,.406],[.229,.224,.225])])(im.convert('RGB'))
            import torch
            torch.testing.assert_close(x[0],expected,rtol=0,atol=0)
            self.assertGreater(float(x[0].std()),.1)

    def test_checkpoint_roundtrip(self):
        import tempfile
        from pathlib import Path
        import torch
        from cwru_bearing.models import build_model
        c=load_config('cross_load');m=build_model(c,pretrained=False)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture.pth'
            torch.save({'model_state':m.state_dict(),'config':c},p)
            loaded=torch.load(p,map_location='cpu',weights_only=True)
            n=build_model(loaded['config'],pretrained=False)
            n.load_state_dict(loaded['model_state'])
            self.assertTrue(all(torch.equal(v,n.state_dict()[k]) for k,v in m.state_dict().items()))

    def test_toy_training_artifact_persistence(self):
        """Two tiny classifier updates test persistence, not CWRU/model performance."""
        import tempfile
        from pathlib import Path
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        from cwru_bearing.training import train_stage
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.features=nn.Sequential(nn.Linear(2,2))
                self.classifier=nn.Linear(2,4)
            def forward(self,x):
                return self.classifier(self.features(x))
        config=load_config('cross_fault_size')
        config={**config,'stage1':{**config['stage1'],'epochs':2}}
        loader=DataLoader(TensorDataset(torch.eye(2).repeat(2,1),torch.arange(4)),batch_size=4)
        with tempfile.TemporaryDirectory() as d:
            model=Toy()
            epoch=train_stage(model,{'train':loader,'val':loader},config,1,'cpu',Path(d))
            checkpoint=torch.load(Path(d)/'stage1_best.pth',weights_only=True)
            self.assertEqual(checkpoint['best_epoch'],epoch)
            self.assertTrue((Path(d)/'stage1_history.csv').is_file())
            self.assertTrue((Path(d)/'stage1_history.json').is_file())
            self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in checkpoint['model_state'].items()))


if __name__ == '__main__':
    unittest.main()

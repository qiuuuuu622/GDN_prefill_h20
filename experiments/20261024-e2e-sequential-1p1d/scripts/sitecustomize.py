"""Process-local import hook; never modifies installed SGLang/FlashInfer."""
import os,sys,importlib.abc,importlib.machinery
if os.environ.get('GDN35_PATCH')=='1':
 class Loader(importlib.abc.Loader):
  def __init__(self,base):self.base=base
  def create_module(self,spec):return self.base.create_module(spec) if hasattr(self.base,'create_module') else None
  def exec_module(self,module):
   self.base.exec_module(module)
   from service_patch import install
   install(module)
 class Finder(importlib.abc.MetaPathFinder):
  def find_spec(self,fullname,path=None,target=None):
   if fullname!='sglang.srt.layers.attention.linear.kernels.gdn_flashinfer':return None
   spec=importlib.machinery.PathFinder.find_spec(fullname,path)
   if spec and spec.loader:spec.loader=Loader(spec.loader)
   return spec
 sys.meta_path.insert(0,Finder())

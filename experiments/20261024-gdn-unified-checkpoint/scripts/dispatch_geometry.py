"""Parameterize the previous conservative CTA estimator by rank-local Hv.

Experimental: retain the historical H20 split-cost/margin, do not claim a generic
hardware autotuner. Call once on CPU batch metadata, not once per layer.
"""
import functools,heapq
@functools.lru_cache(maxsize=1024)
def choose(lengths,hv,sms=78):
 if not lengths or hv<=0 or sms<=0 or any(n<0 for n in lengths):raise ValueError('invalid geometry')
 if len(lengths)==1:return 'dv'
 def makespan(heads):
  slots=[0]*sms
  for n in lengths:
   if n:
    for _ in range(heads):heapq.heapreplace(slots,slots[0]+((n+63)//64))
  return max(slots)
 return 'dv' if .67*makespan(2*hv)<.85*makespan(hv) else 'full'

"""Conservative CPU-side CTA load estimate, computed once per batch metadata."""
import functools,heapq
@functools.lru_cache(maxsize=1024)
def method_for_lengths(lengths):
 if len(lengths)==1:return 'dv'
 def makespan(heads):
  sm=[0]*78
  for n in lengths:
   if n==0:continue
   for _ in range(heads):heapq.heapreplace(sm,sm[0]+((n+63)//64))
  return max(sm)
 # Full-V versus split-V per-tile pilot ratio. Require a 15% margin before
 # using split-V for multi-sequence calls; otherwise keep full-V arithmetic.
 return 'dv' if .67*makespan(64)<.85*makespan(32) else 'full'

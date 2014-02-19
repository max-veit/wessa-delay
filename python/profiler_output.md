=Generating code (IPython)=
    # Cell
    ens = we.Ensemble(2, paving, (10, 15), init_trjs)
    niter = 50
    pbar = ProgressBar(niter)

    # Cell
    %%prun
    for idx in range(niter):
        ens.run_step()
        pbar.animate(idx + 1)

=Output=

         30464014 function calls in 106.022 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
  3152988   17.896    0.000   17.896    0.000 ssad.py:75(calc_propensity)
  1576494   17.151    0.000   48.761    0.000 ssad.py:257(_sample_next_reaction)
  1592204   13.394    0.000   37.367    0.000 ssad.py:241(_can_run_rxn)
  1592204   11.112    0.000   11.112    0.000 {method 'reduce' of 'numpy.ufunc' objects}
  1576494    8.622    0.000    9.831    0.000 ssad.py:288(_execute_rxn)
  1576494    6.163    0.000    6.163    0.000 {method 'cumsum' of 'numpy.ndarray' objects}
    15710    5.695    0.000  101.668    0.006 ssad.py:169(run_dynamics)
  1592204    4.691    0.000    4.691    0.000 {built-in method empty}
  1612175    3.380    0.000    3.380    0.000 {built-in method array}
  1592204    3.169    0.000   23.973    0.000 fromnumeric.py:1842(all)
  1592204    2.319    0.000   13.431    0.000 _methods.py:35(_all)
  1592204    2.175    0.000    5.507    0.000 numeric.py:462(asanyarray)
  1592204    1.865    0.000   15.296    0.000 {method 'all' of 'numpy.ndarray' objects}
  1576494    1.272    0.000    7.436    0.000 fromnumeric.py:1915(cumsum)
  3157399    1.211    0.000    1.211    0.000 {method 'append' of 'list' objects}
     4261    1.091    0.000    2.191    0.001 ensemble.py:75(clone)
      434    1.047    0.002    1.126    0.003 ensemble.py:306(_reduce_bin)
  1576494    0.979    0.000    0.979    0.000 {method 'exponential' of 'mtrand.RandomState' objects}
     4261    0.712    0.000    0.744    0.000 ssad.py:138(__init__)
    15710    0.366    0.000    0.557    0.000 ensemble.py:408(get_bin_num)
  1576494    0.359    0.000    0.359    0.000 {method 'random_sample' of 'mtrand.RandomState' objects}
     4261    0.324    0.000    1.068    0.000 ensemble.py:53(__init__)
  1589221    0.299    0.000    0.299    0.000 {built-in method len}
       50    0.137    0.003    3.593    0.072 ensemble.py:298(_resample)
    15710    0.089    0.000    0.089    0.000 {built-in method ravel_multi_index}
    95941    0.072    0.000    0.072    0.000 ensemble.py:141(__lt__)
       50    0.066    0.001    0.672    0.013 ensemble.py:256(_recompute_bins)
    23931    0.055    0.000    0.083    0.000 {built-in method heappush}
       50    0.039    0.001  101.720    2.034 ensemble.py:287(_run_dynamics_all)
      671    0.035    0.000    2.329    0.003 ensemble.py:320(_grow_bin)
     7920    0.034    0.000    0.065    0.000 {built-in method heappop}
    20463    0.033    0.000    0.052    0.000 functools.py:85(<lambda>)
    19971    0.030    0.000    0.079    0.000 numeric.py:392(asarray)
     4311    0.021    0.000    0.073    0.000 {built-in method max}
    31520    0.019    0.000    0.019    0.000 ensemble.py:282(__iter__)
     8522    0.015    0.000    0.028    0.000 numeric.py:1810(isscalar)
    15710    0.014    0.000    0.014    0.000 ssad.py:297(_save_run_state)
    10022    0.014    0.000    0.014    0.000 {built-in method isinstance}
    15710    0.009    0.000    0.009    0.000 {method 'transpose' of 'numpy.ndarray' objects}
    15153    0.006    0.000    0.006    0.000 ensemble.py:144(__eq__)
       50    0.003    0.000    0.006    0.000 uuid.py:554(uuid4)
      200    0.002    0.000    0.003    0.000 encoder.py:196(iterencode)
     8522    0.002    0.000    0.002    0.000 ssad.py:162(<lambda>)
      350    0.002    0.000    0.002    0.000 {method 'send' of 'zmq.backend.cython.socket.Socket' objects}
       50    0.002    0.000    0.002    0.000 __init__.py:49(create_string_buffer)
       50    0.001    0.000    0.001    0.000 uuid.py:104(__init__)
      200    0.001    0.000    0.006    0.000 __init__.py:187(dumps)
       50    0.001    0.000    0.001    0.000 {built-in method now}
       50    0.001    0.000    0.026    0.001 session.py:555(send)
      200    0.001    0.000    0.002    0.000 iostream.py:178(write)
       50    0.001    0.000    0.001    0.000 <ipython-input-15-a0c4d9415c3f>:19(__update_amount)
       50    0.001    0.000    0.030    0.001 iostream.py:122(flush)
      200    0.001    0.000    0.007    0.000 jsonapi.py:46(dumps)
      600    0.001    0.000    0.001    0.000 traitlets.py:282(__get__)
       50    0.001    0.000    0.001    0.000 {zmq.backend.cython._poll.zmq_poll}
       50    0.001    0.000    0.011    0.000 session.py:498(serialize)
      200    0.001    0.000    0.004    0.000 encoder.py:175(encode)
       50    0.001    0.000    0.003    0.000 {built-in method print}
        1    0.001    0.001  106.022  106.022 <string>:2(<module>)
       50    0.001    0.000  105.985    2.120 ensemble.py:220(run_step)
       50    0.001    0.000    0.003    0.000 socket.py:210(send_multipart)
       50    0.001    0.000    0.001    0.000 {method 'isoformat' of 'datetime.datetime' objects}
       50    0.001    0.000    0.036    0.001 <ipython-input-15-a0c4d9415c3f>:10(animate)
       50    0.001    0.000    0.003    0.000 session.py:483(sign)
       50    0.001    0.000    0.002    0.000 <ipython-input-15-a0c4d9415c3f>:15(update_iteration)
     1322    0.001    0.000    0.001    0.000 ensemble.py:263(<lambda>)
       50    0.001    0.000    0.010    0.000 session.py:464(msg)
      200    0.001    0.000    0.001    0.000 encoder.py:98(__init__)
       50    0.000    0.000    0.002    0.000 iostream.py:105(_flush_from_subprocesses)
       50    0.000    0.000    0.001    0.000 attrsettr.py:47(__getattr__)
      200    0.000    0.000    0.007    0.000 session.py:83(<lambda>)
       50    0.000    0.000    0.009    0.000 session.py:461(msg_header)
      250    0.000    0.000    0.000    0.000 {method 'update' of '_hashlib.HASH' objects}
       50    0.000    0.000    0.000    0.000 uuid.py:230(__str__)
       50    0.000    0.000    0.007    0.000 session.py:426(msg_id)
      300    0.000    0.000    0.000    0.000 iostream.py:79(_is_master_process)
      100    0.000    0.000    0.001    0.000 {built-in method hasattr}
       50    0.000    0.000    0.000    0.000 session.py:196(extract_header)
      250    0.000    0.000    0.001    0.000 iostream.py:88(_check_mp_mode)
       50    0.000    0.000    0.001    0.000 hmac.py:82(copy)
      250    0.000    0.000    0.000    0.000 {method 'encode' of 'str' objects}
      100    0.000    0.000    0.000    0.000 {built-in method round}
      200    0.000    0.000    0.001    0.000 hmac.py:75(update)
       50    0.000    0.000    0.001    0.000 iostream.py:82(_is_master_thread)
       50    0.000    0.000    0.001    0.000 poll.py:86(poll)
      150    0.000    0.000    0.000    0.000 {method 'copy' of '_hashlib.HASH' objects}
       50    0.000    0.000    0.000    0.000 iostream.py:218(_new_buffer)
       50    0.000    0.000    0.001    0.000 session.py:192(msg_header)
       50    0.000    0.000    0.000    0.000 session.py:618(<listcomp>)
       50    0.000    0.000    0.001    0.000 iostream.py:209(_flush_buffer)
       50    0.000    0.000    0.000    0.000 {built-in method getattr}
       50    0.000    0.000    0.000    0.000 threading.py:1204(current_thread)
       50    0.000    0.000    0.000    0.000 {method 'getvalue' of '_io.StringIO' objects}
       50    0.000    0.000    0.000    0.000 <ipython-input-15-a0c4d9415c3f>:28(__str__)
      200    0.000    0.000    0.000    0.000 {built-in method time}
       50    0.000    0.000    0.001    0.000 jsonutil.py:94(date_default)
      200    0.000    0.000    0.000    0.000 {method 'write' of '_io.StringIO' objects}
       50    0.000    0.000    0.000    0.000 {method 'digest' of '_hashlib.HASH' objects}
       50    0.000    0.000    0.000    0.000 hmac.py:95(_current)
       50    0.000    0.000    0.000    0.000 {built-in method locals}
       50    0.000    0.000    0.000    0.000 {method 'hexdigest' of '_hashlib.HASH' objects}
       50    0.000    0.000    0.000    0.000 hmac.py:114(hexdigest)
       50    0.000    0.000    0.000    0.000 threading.py:1055(ident)
       50    0.000    0.000    0.000    0.000 {built-in method iter}
      350    0.000    0.000    0.000    0.000 {built-in method getpid}
       50    0.000    0.000    0.000    0.000 py3compat.py:20(encode)
      150    0.000    0.000    0.000    0.000 {method 'items' of 'dict' objects}
        1    0.000    0.000  106.022  106.022 {built-in method exec}
      200    0.000    0.000    0.000    0.000 {method 'join' of 'str' objects}
       50    0.000    0.000    0.000    0.000 {method 'count' of 'list' objects}
       50    0.000    0.000    0.000    0.000 {built-in method __new__ of type object at 0x7f0aeab6a400}
       50    0.000    0.000    0.000    0.000 {method 'copy' of 'dict' objects}
       50    0.000    0.000    0.000    0.000 {built-in method get_ident}
       50    0.000    0.000    0.000    0.000 {method 'close' of '_io.StringIO' objects}
       50    0.000    0.000    0.000    0.000 {method 'upper' of 'str' objects}
      100    0.000    0.000    0.000    0.000 {method 'extend' of 'list' objects}
       50    0.000    0.000    0.000    0.000 {method 'get' of 'dict' objects}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}

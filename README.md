CUDA_Dynamic_Programming_Example_3
==================================

Yet another 64 bit double precision DP problem adapted to CUDA(someone has to do it)..

CUDA adaptation of the Top Coder Division I problem:

http://community.topcoder.com/stat?c=problem_statement&pm=10771&rd=14146

The running time of this implemenation is 2*((num cities+1)*(num visits +1)*((num cities*(max possible fans))))+(num cities*(max possible fans))), so for the larger example in table below 2*(101*61*(101*101))+(101*101)= 125,706,923. It uses slightly more memory than that, just to be safe. There are other implementations which use less memory, so email me if you would like that implementation.

The larger the data set, the more the CUDA implemenation outperforms the serial CPU version. If the compute capability of your GPU is less than 3.5, cast to 32 bit floating point.

____
<table>
<tr>
    <th>Num Cities</th><th>Visits(K)</th><th>Max Fans</th><th>CPU time</th><th>GPU time</th><th>CUDA Speedup</th>
</tr>

  <tr>
    <td>36</td><td>21</td><td>40</td><td> 34 ms</td><td>  3 ms</td><td> 11.0x</td>
  </tr>
  <tr>
    <td>100</td><td>60 </td><td>100</td><td> 5216 ms</td><td>  101 ms</td><td> 51.64x</td>
  </tr>
</table>  
___

NOTE: All CUDA GPU times include all device memsets, host-device memory copies and device-host memory copies.

CPU= Intel I-7 3770K 3.5 Ghz with 3.9 Ghz target

GPU= Tesla K20c 5GB

Windows 7 Ultimate x64

Visual Studio 2010 x64

Would love to see a faster Python version, since that is the *best* language these days. Please contact me with the running time for the same sample sizes!

Python en Ruby zijn talen voor de lui en traag!  

Python und Ruby sind Sprachen für die faul und langsam!  

Python et Ruby sont des langues pour les paresseux et lent!  


<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','//www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-43459430-1', 'github.com');
  ga('send', 'pageview');

</script>

[![githalytics.com alpha](https://cruel-carlota.pagodabox.com/d40d1ae4136dd45569d36b3e67930e12 "githalytics.com")](http://githalytics.com/OlegKonings/CUDA_vs_CPU_DynamicProgramming_double)
[![githalytics.com alpha](https://cruel-carlota.pagodabox.com/d40d1ae4136dd45569d36b3e67930e12 "githalytics.com")](http://githalytics.com/OlegKonings/CUDA_vs_CPU_DynamicProgramming_double)

# split the results

# 2305

# predio
python split_results.py -n cf -t predio -m mean5#1 -b bags/2305/2019-05-23-16-05-59.bag -s 1558638359 -ts 408 808
python split_results.py -n cf -t predio -m mean5#2 -b bags/2305/2019-05-23-16-27-17.bag -s 1558639638 -ts 31 88
python split_results.py -n cf -t predio -m net5 -b bags/2305/2019-05-23-16-27-17.bag -s 1558639638 -ts 118 527
python split_results.py -n ls -t predio -m net5 -b bags/2305/2019-05-23-16-27-17.bag -s 1558639638 -ts 728 1110
python split_results.py -n ls -t predio -m mean5 -b bags/2305/2019-05-23-16-27-17.bag -s 1558639638 -ts 1126 1505
python split_results.py -n jf -t predio -m net20 -b bags/2305/2019-05-23-16-27-17.bag -s 1558639638 -ts 1981 2333
python split_results.py -n jf -t predio -m mean20 -b bags/2305/2019-05-23-16-27-17.bag -s 1558639638 -ts 2374 2809

# oito
python split_results.py -n cf -t oito -m mean5 -b bags/2305/2019-05-23-16-27-17.bag -s 1558639638 -ts 3039 3191 3295 3437
python split_results.py -n cf -t oito -m net5 -b bags/2305/2019-05-23-16-27-17.bag -s 1558639638 -ts 3465 3621 3643 3788
python split_results.py -n ls -t oito -m net5 -b bags/2305/2019-05-23-16-27-17.bag -s 1558639638 -ts 3929 4072 4093 4237
python split_results.py -n ls -t oito -m mean5 -b bags/2305/2019-05-23-16-27-17.bag -s 1558639638 -ts 4267 4446 4461 4622
python split_results.py -n jf -t oito -m net20 -b bags/2305/2019-05-23-16-27-17.bag -s 1558639638 -ts 4706 4859 4928 5114
python split_results.py -n jf -t oito -m mean20 -b bags/2305/2019-05-23-16-27-17.bag -s 1558639638 -ts 5136 5273 5336 5472

# 2405 manha

# predio
python split_results.py -n tg -t predio -m net5 -b bags/2405/2019-05-24-11-01-21.bag -s 1558706481 -ts 1085 1462
python split_results.py -n tg -t predio -m net20 -b bags/2405/2019-05-24-11-01-21.bag -s 1558706481 -ts 1611 1989
python split_results.py -n ed -t predio -m net5 -b bags/2405/2019-05-24-12-15-07.bag -s 1558710908 -ts 1370 1753
python split_results.py -n ed -t predio -m mean5 -b bags/2405/2019-05-24-12-15-07.bag -s 1558710908 -ts 1797 2166
python split_results.py -n kl -t predio -m mean20 -b bags/2405/2019-05-24-12-15-07.bag -s 1558710908 -ts 81 469
python split_results.py -n kl -t predio -m net20 -b bags/2405/2019-05-24-12-15-07.bag -s 1558710908 -ts 565 948

# oito
python split_results.py -n tg -t oito -m net5 -b bags/2405/2019-05-24-11-01-21.bag -s 1558706481 -ts 2306 2456 2485 2638
python split_results.py -n tg -t oito -m net20 -b bags/2405/2019-05-24-11-01-21.bag -s 1558706481 -ts 2679 2831 2853 2990
python split_results.py -n ed -t oito -m net5 -b bags/2405/2019-05-24-12-15-07.bag -s 1558710908 -ts 2390 2535 2549 2700
python split_results.py -n ed -t oito -m mean5 -b bags/2405/2019-05-24-12-15-07.bag -s 1558710908 -ts 2728 2896 2958 3128
python split_results.py -n kl -t oito -m mean20 -b bags/2405/2019-05-24-12-15-07.bag -s 1558710908 -ts 3209 3332 3353 3480
python split_results.py -n kl -t oito -m net20 -b bags/2405/2019-05-24-12-15-07.bag -s 1558710908 -ts 3568 3697 3712 3845

# 2405 tarde

# predio
python split_results.py -n ck1 -t predio -m net20 -b bags/2405/2019-05-24-14-47-23.bag -s 1558720043 -ts 279 460
python split_results.py -n ck1 -t predio -m mean20 -b bags/2405/2019-05-24-14-47-23.bag -s 1558720043 -ts 626 686
python split_results.py -n gb -t predio -m net20 -b bags/2405/2019-05-24-16-14-07.bag -s 1558725247 -ts 876 1255
python split_results.py -n gb -t predio -m net5 -b bags/2405/2019-05-24-16-14-07.bag -s 1558725247 -ts 1267 1640
python split_results.py -n sk -t predio -m mean20 -b bags/2405/2019-05-24-17-39-07.bag -s 1558730348 -ts 376 759
python split_results.py -n sk -t predio -m net20 -b bags/2405/2019-05-24-17-39-07.bag -s 1558730348 -ts 780 1154

# oito
python split_results.py -n ck1 -t oito -m net5 -b bags/2405/2019-05-24-14-47-23.bag -s 1558720043 -ts 765 865 877 984
python split_results.py -n ck1 -t oito -m mean5 -b bags/2405/2019-05-24-14-47-23.bag -s 1558720043 -ts 1016 1107 1122 1215
python split_results.py -n gb -t oito -m net20 -b bags/2405/2019-05-24-16-14-07.bag -s 1558725247 -ts 249 348 378 478
python split_results.py -n gb -t oito -m net5 -b bags/2405/2019-05-24-16-14-07.bag -s 1558725247 -ts 504 645 656 764
python split_results.py -n sk -t oito -m mean20 -b bags/2405/2019-05-24-17-39-07.bag -s 1558730348 -ts 1273 1397 1435 1592
python split_results.py -n sk -t oito -m net20 -b bags/2405/2019-05-24-17-39-07.bag -s 1558730348 -ts 1619 1731 1746 1851

# 2505

# predio
python split_results.py -n lf -t predio -m mean5 -b bags/2505/2019-05-25-13-45-40.bag -s 1558802741 -ts 376 750
python split_results.py -n lf -t predio -m mean20 -b bags/2505/2019-05-25-13-45-40.bag -s 1558802741 -ts 762 1134
python split_results.py -n gr -t predio -m net5 -b bags/2505/2019-05-25-13-45-40.bag -s 1558802741 -ts 3814 4188
python split_results.py -n gr -t predio -m mean5 -b bags/2505/2019-05-25-13-45-40.bag -s 1558802741 -ts 4205 4583
python split_results.py -n gm -t predio -m mean5 -b bags/2505/2019-05-25-13-45-40.bag -s 1558802741 -ts 4597 4966
python split_results.py -n gm -t predio -m net5 -b bags/2505/2019-05-25-13-45-40.bag -s 1558802741 -ts 4973 5180

# oito
python split_results.py -n lf -t oito -m mean5 -b bags/2505/2019-05-25-13-45-40.bag -s 1558802741 -ts 1369 1501 1517 1619
python split_results.py -n lf -t oito -m mean20 -b bags/2505/2019-05-25-13-45-40.bag -s 1558802741 -ts 1645 1774 1786 1932
python split_results.py -n gr -t oito -m net5 -b bags/2505/2019-05-25-13-45-40.bag -s 1558802741 -ts 2342 2465 2491 2607
python split_results.py -n gr -t oito -m mean5 -b bags/2505/2019-05-25-13-45-40.bag -s 1558802741 -ts 2679 2785 2801 2906
python split_results.py -n gm -t oito -m mean5 -b bags/2505/2019-05-25-13-45-40.bag -s 1558802741 -ts 3003 3146 3187 3295
python split_results.py -n gm -t oito -m net5 -b bags/2505/2019-05-25-13-45-40.bag -s 1558802741 -ts 3316 3465 3486 3639

# 2705

# predio
python split_results.py -n am -t predio -m mean20 -b bags/2705/2019-05-27-11-47-59.bag -s 1558968479 -ts 1237 1432
python split_results.py -n am -t predio -m mean5 -b bags/2705/2019-05-27-11-47-59.bag -s 1558968479 -ts 1445 1642
python split_results.py -n am -t predio -m off -b bags/2705/2019-05-27-11-47-59.bag -s 1558968479 -ts 1662 1863

# oito
python split_results.py -n am -t oito -m mean20 -b bags/2705/2019-05-27-11-47-59.bag -s 1558968479 -ts 2331 2454 2468 2607
python split_results.py -n am -t oito -m mean5 -b bags/2705/2019-05-27-11-47-59.bag -s 1558968479 -ts 2643 2780 2807 2908
python split_results.py -n am -t oito -m off -b bags/2705/2019-05-27-11-47-59.bag -s 1558968479 -ts 2928 3030 3041 3112 

# 2805

# predio
python split_results.py -n ck2 -t predio -m off -b bags/2805/2019-05-28-09-13-38.bag -s 1559045618 -ts 377 556
python split_results.py -n ck2 -t predio -m mean5 -b bags/2805/2019-05-28-09-13-38.bag -s 1559045618 -ts 584 765
python split_results.py -n ck2 -t predio -m net5 -b bags/2805/2019-05-28-09-13-38.bag -s 1559045618 -ts 790 972	

# oito
python split_results.py -n ck2 -t oito -m off -b bags/2805/2019-05-28-09-13-38.bag -s 1559045618 -ts 1074 1164 1074 1164
python split_results.py -n ck2 -t oito -m mean5 -b bags/2805/2019-05-28-09-13-38.bag -s 1559045618 -ts 1184 1286 1184 1286
python split_results.py -n ck2 -t oito -m net5 -b bags/2805/2019-05-28-09-13-38.bag -s 1559045618 -ts 1319 1415 1319 1415

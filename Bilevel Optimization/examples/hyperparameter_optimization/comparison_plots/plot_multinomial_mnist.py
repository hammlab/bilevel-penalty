# -*- coding: utf-8 -*-
"""
Created on Mon May 21 21:30:36 2018

@author: Akshay
"""

import numpy as np
import matplotlib.pyplot as plt


time_our = [0.15, 0.187836885452, 0.327199888229, 0.465827083588, 0.607615566254, 0.738769721985, 0.880207920074, 1.02585849762, 1.17248215675, 1.31007814407, 1.45528492928, 1.60003685951, 1.73802204132, 1.87661385536, 2.01895704269, 2.15899882317, 2.30660524368, 2.44605469704, 2.59181594849, 2.73105573654, 2.88172049522, 3.02923703194, 3.16995940208, 3.3078915596, 3.45167379379, 3.59611344337, 3.73483991623, 3.87724032402, 4.01584291458, 4.16100435257, 4.30115194321, 4.45378909111, 4.59289631844, 4.73110671043, 4.87572565079, 5.02175445557, 5.16776189804, 5.31022667885, 5.44761309624, 5.59088134766, 5.72831215858, 5.87348256111, 6.01635565758, 6.15847678185, 6.30089559555, 6.44798660278, 6.59009714127, 6.73016195297, 6.87803740501, 7.02707462311, 7.16980805397, 7.31137442589, 7.44780020714, 7.59791946411, 7.73385391235, 7.87130823135, 8.01176028252, 8.14812932014, 8.29368114471, 8.44343118668, 8.57972660065, 8.73396782875, 8.87766680717, 9.01881985664, 9.15204043388, 9.28382668495, 9.41939148903, 9.5633749485, 9.70775332451, 9.84461688995, 9.98944454193, 10.1282017231, 10.2687552452, 10.4110422611, 10.5612578869, 10.7034118176, 10.8429780483, 10.9840176582, 11.1217888355, 11.2675624371, 11.4148589134, 11.5521996975, 11.6830717087, 11.820499897, 11.9607467175, 12.1002926826, 12.2419544697, 12.3906906605, 12.5227347851, 12.6633383751, 12.7987145901, 12.9439013481, 13.0844331264, 13.223768425, 13.3656516075, 13.5100084305, 13.6478804588, 13.7970629215, 13.9411358356, 14.0849123478, 14.2267119884, 14.3736151695, 14.5131673336, 14.6546535492, 14.7981531143, 14.9409342766, 15.0779576302, 15.2155883789, 15.3620213509, 15.5069495678, 15.653826952, 15.7972865582, 15.942518568, 16.0836967945, 16.2227558136, 16.3672660351, 16.5075540543, 16.6411642075, 16.7780357361, 16.9140419483, 17.0573130608, 17.201065731, 17.3372461319, 17.4796415329, 17.620960474, 17.7579905987, 17.9011284828, 18.0360954285, 18.1839041233, 18.3239063263, 18.4687441826, 18.6159248829, 18.7620715141, 18.9102041245, 19.0528256416, 19.198960638, 19.3449222088, 19.4930979729, 19.6423826218, 19.784115839, 19.9186247826, 20.0516188145, 20.1860014439, 20.3237186432, 20.4661617756, 20.6115660191, 20.7442844391, 20.8839346409, 21.0291904926, 21.166556263, 21.304992485, 21.4515116692, 21.5944994926, 21.73630476, 21.8787635326, 22.0173945427, 22.1627411366, 22.3087051392, 22.4504821301, 22.5878959656, 22.7219691753, 22.8606715202, 23.004016304, 23.1463099003, 23.2850497246, 23.4204715729, 23.5579425335, 23.696550703, 23.8448173046, 23.982320118, 24.1201399326, 24.2654465199, 24.4129605293, 24.5614694595, 24.707897377, 24.853442955, 24.9955313683, 25.1362833977, 25.2763954163, 25.4214405537, 25.5544819355, 25.7008641243, 25.8436781406, 25.978399086, 26.1125211239, 26.2551341534, 26.3936805248, 26.5303105831, 26.6695077419, 26.8075813293, 26.9583187103, 27.1032227516, 27.2486629486, 27.3880087376, 27.5328257084, 27.6797069073, 27.8283698559, 27.9731784821, 28.1184190273, 28.2572080135, 28.3968637943, 28.5369936466, 28.6769543171, 28.8190121174, 28.9650361061, 29.1124361038, 29.2487300873, 29.3919848919, 29.5267067432, 29.6685061932, 29.8093716145, 29.9473229885, 30.0844879627, 30.2260483742, 30.3679079533, 30.5055315495, 30.6435339451, 30.7816969395, 30.9222326756, 31.0686598778, 31.2179802418, 31.3627902985, 31.506732893, 31.6428851604, 31.7811353683, 31.9236519814, 32.0626411438, 32.2048335552, 32.344852829, 32.4852430344, 32.6232628345, 32.7622820854, 32.9118067265, 33.0516849041, 33.187254858, 33.3217229843, 33.4594719887, 33.6057298183, 33.7515899181, 33.9009012699, 34.0430384159, 34.1901936054, 34.3283249855, 34.4757061958, 34.6183480263, 34.7547791481, 34.900104332, 35.0394588947, 35.1778788567, 35.3208399296, 35.4545585155, 35.5947896957, 35.7341002464, 35.8784802914, 36.0203885078, 36.1646909237, 36.306347084, 36.4561444759, 36.5965562344, 36.7426900387, 36.8867497921, 37.0329467773, 37.1794352055, 37.3241440296, 37.4692848682, 37.6143320084, 37.7515964031, 37.8942024231, 38.0361404419, 38.1748408794, 38.3172441006, 38.4521574497, 38.5898815155, 38.7234687805, 38.8651051044, 39.0078232288, 39.1504332066, 39.2941278458, 39.440763855, 39.5824504375, 39.7175190926, 39.8553096771, 40.0064998627, 40.1475543022, 40.2889240742, 40.43355093, 40.5707904816, 40.7169448853, 40.8554715157, 40.9984922409, 41.134844017, 41.2714160919, 41.4091272831, 41.5460271358, 41.6831963062, 41.8257871151, 41.9655302048, 42.1119142056, 42.2460594654, 42.3846423149, 42.5296384811, 42.6707752705, 42.8186894894, 42.9640918732, 43.1037234783, 43.2454284668, 43.3816550255, 43.5176096439, 43.6539008617, 43.7957252502, 43.9423168659, 44.0797934532, 44.2225808144, 44.3702778339, 44.5117870331, 44.6480428219, 44.789155817, 44.9336408615, 45.0745724201, 45.2163883686, 45.3541619301, 45.5006544113, 45.6424525738, 45.7831996918, 45.917937851, 46.0618731976, 46.2150212765, 46.3529117584, 46.4955993176, 46.6369243622, 46.7814039707, 46.9163618088, 47.057839632, 47.1945564747, 47.3415490627, 47.4840710163, 47.6211131096, 47.7616373539, 47.9024209499, 48.0436423779, 48.1837316036, 48.3238598347, 48.4652942181, 48.6095984459, 48.7499244213, 48.8935302258, 49.0323147774, 49.1782177925, 49.3229511738, 49.4657539845, 49.6040083408, 49.7410307407, 49.8852353096, 50.0233409882, 50.1703809738, 50.3183903694, 50.4713269234, 50.6099267483, 50.7505717278, 50.8893799305, 51.0280873775, 51.1806485176, 51.3281213284, 51.4752815247, 51.6185514927, 51.7587246418, 51.9077714443, 52.0550606728, 52.1990962505, 52.3490826607, 52.4869140148, 52.6290803432, 52.7730911732, 52.9171533585, 53.0611074924, 53.2153162956, 53.3603085518, 53.4998865128, 53.6467794418, 53.7833231926, 53.9234613895, 54.0634682655, 54.2115308285, 54.3510538578, 54.4955574989, 54.6427699089, 54.7850410938, 54.9309160709, 55.0792308807, 55.22555933, 55.375707531, 55.5145607471, 55.6672547817, 55.8108373642, 55.9596078873, 56.0965737343, 56.2288858891, 56.3724856853, 56.5089391232, 56.654198122, 56.7928620815, 56.9293465137, 57.0724195004, 57.218221283, 57.3579555988, 57.4932077885, 57.6326146126, 57.7746665001, 57.912942791, 58.0481624126, 58.1848688602, 58.3252188683, 58.464408493, 58.5951906681, 58.7334334373, 58.8719794273, 59.0116948605, 59.1456930637, 59.290871048, 59.4315104485, 59.5752090931, 59.7156917095, 59.8537259579, 59.9898913383, 60.1212385178, 60.2571629047, 60.3935577869, 60.5298037529, 60.6667665005, 60.805191803, 60.9445065975, 61.0834210873, 61.2210000992, 61.3590085983, 61.4920566559, 61.6270190239, 61.772651577, 61.9090222359, 62.0502487183, 62.1906026363, 62.3391366959, 62.4774579048, 62.6202575207, 62.7574857235, 62.897642374, 63.0408081055, 63.1883113384, 63.3318698883, 63.4707724094, 63.6185054302, 63.7683790207, 63.9093802452, 64.0487992287, 64.190506649, 64.3282091618, 64.4711270332, 64.6040052414, 64.7542916298, 64.9018481731, 65.0451059341, 65.1806414127, 65.3226242542, 65.4631924629, 65.6014984608, 65.7354784489, 65.8767473221, 66.0216341496, 66.1641322613, 66.3016918659, 66.4446168423, 66.5872027397, 66.7293725014, 66.8825407028, 67.0258659363, 67.1616889477, 67.2981732845, 67.4428952694, 67.5790999413, 67.7205453396, 67.8642762661, 68.0092918873, 68.1568097591, 68.3037713528, 68.4447062016, 68.5986444473, 68.7380238056, 68.8797092438, 69.0198848248, 69.1617946148, 69.2989010334, 69.4376085281, 69.5806044579, 69.7239285946, 69.8620690346, 69.9988116741, 70.1390515327, 70.2806303501, 70.4269678116, 70.5687232494, 70.7111072063, 3356]
loss_our = [20000, 19344.4464844, 16287.0199219, 14058.478125, 12285.2060547, 10697.6703125, 9589.54785156, 8798.91132812, 8103.21894531, 7458.36240234, 6937.54433594, 6547.21464844, 6223.09580078, 5961.45664062, 5747.5984375, 5547.40019531, 5357.48515625, 5198.09882813, 5066.07294922, 4946.46435547, 4847.16474609, 4755.99902344, 4667.24697266, 4588.95664062, 4522.31337891, 4463.21777344, 4406.09091797, 4354.03398437, 4305.59003906, 4262.44023437, 4222.06816406, 4182.69355469, 4147.97109375, 4110.62548828, 4075.38930664, 4045.26601562, 4014.5871582, 3989.51396484, 3959.25048828, 3937.84624023, 3912.83916016, 3896.40698242, 3868.40854492, 3850.82797852, 3831.45771484, 3816.54394531, 3793.40356445, 3776.41098633, 3763.6015625, 3743.61464844, 3726.65883789, 3711.21894531, 3700.19667969, 3682.78271484, 3666.95419922, 3655.93320313, 3646.01679688, 3630.08613281, 3617.07133789, 3607.5871582, 3597.74672852, 3585.38608398, 3573.50302734, 3565.00512695, 3555.79125977, 3547.51259766, 3542.78071289, 3545.12392578, 3526.96855469, 3524.80366211, 3540.06640625, 3513.27348633, 3525.19501953, 3482.04941406, 3500.08378906, 3472.59770508, 3479.91357422, 3474.05913086, 3462.37788086, 3480.29609375, 3447.58725586, 3442.36513672, 3438.01342773, 3418.82397461, 3426.31967773, 3418.44482422, 3400.01499023, 3394.84677734, 3386.94477539, 3386.31748047, 3379.56435547, 3367.6065918, 3364.43227539, 3361.62680664, 3357.36655273, 3352.83149414, 3347.86499023, 3341.93774414, 3337.43647461, 3334.37373047, 3328.75527344, 3323.92631836, 3321.11337891, 3319.14072266, 3316.40385742, 3311.13510742, 3306.38164062, 3303.03222656, 3302.03847656, 3300.09033203, 3295.11137695, 3290.67573242, 3287.70419922, 3284.89555664, 3282.80800781, 3280.5644043, 3277.1609375, 3273.88579102, 3271.38896484, 3268.70444336, 3266.12050781, 3263.41738281, 3260.33291016, 3257.56713867, 3254.95493164, 3252.42402344, 3250.50166016, 3248.14135742, 3245.63427734, 3242.77182617, 3242.86611328, 3243.82553711, 3252.74628906, 3240.34326172, 3251.01879883, 3239.82714844, 3271.6324707, 3274.14462891, 3263.13891602, 3259.69755859, 3243.00488281, 3247.50776367, 3244.07949219, 3246.16704102, 3224.94990234, 3225.20415039, 3232.82211914, 3228.86479492, 3221.61948242, 3213.66801758, 3208.14575195, 3211.80087891, 3216.36494141, 3209.17983398, 3199.06274414, 3195.34008789, 3195.33417969, 3198.09838867, 3196.71108398, 3194.58100586, 3187.20019531, 3187.47006836, 3184.62558594, 3185.50371094, 3184.06098633, 3183.42255859, 3175.60234375, 3174.02646484, 3174.34887695, 3174.47368164, 3172.06240234, 3169.40810547, 3167.61899414, 3166.31992188, 3164.02993164, 3163.49365234, 3162.6184082, 3160.40756836, 3157.02446289, 3155.91298828, 3155.52949219, 3154.5480957, 3152.04023438, 3149.92382812, 3148.66806641, 3147.84077148, 3146.16953125, 3144.5565918, 3143.28173828, 3142.06943359, 3140.3137207, 3138.8878418, 3137.87373047, 3136.80864258, 3135.06201172, 3133.67729492, 3132.74296875, 3131.46152344, 3130.29506836, 3128.25180664, 3136.21518555, 3141.75693359, 3159.50239258, 3166.20727539, 3128.29306641, 3146.10654297, 3123.01650391, 3133.11489258, 3145.29916992, 3122.04389648, 3116.90395508, 3120.10639648, 3118.38134766, 3137.87001953, 3181.13217773, 3139.89916992, 3143.42670898, 3144.75136719, 3161.57138672, 3173.09638672, 3132.1777832, 3140.01767578, 3120.77275391, 3130.48061523, 3130.70585937, 3111.41337891, 3111.69389648, 3109.96616211, 3110.76503906, 3138.77148438, 3143.24331055, 3138.34472656, 3121.84052734, 3113.43681641, 3114.97905273, 3120.8597168, 3111.8765625, 3114.62402344, 3105.91855469, 3102.41743164, 3106.74492187, 3101.97988281, 3101.79199219, 3106.80166016, 3100.38920898, 3097.41401367, 3101.78642578, 3102.03833008, 3096.94726562, 3094.97094727, 3093.02797852, 3093.02788086, 3091.57978516, 3087.39819336, 3085.47285156, 3087.75288086, 3088.95654297, 3086.61796875, 3084.98515625, 3083.43144531, 3080.88876953, 3078.14169922, 3077.03881836, 3077.2440918, 3077.8921875, 3078.20395508, 3082.28608398, 3100.10268555, 3151.63959961, 3119.68974609, 3106.39521484, 3116.57451172, 3084.02241211, 3120.96606445, 3117.39262695, 3088.46782227, 3084.47729492, 3089.95073242, 3080.34458008, 3091.04140625, 3097.93266602, 3091.27607422, 3087.75449219, 3086.62929687, 3082.71357422, 3077.94414062, 3079.61005859, 3092.0296875, 3098.54726562, 3093.66342773, 3086.90258789, 3082.76098633, 3080.61210937, 3077.92651367, 3074.63989258, 3075.14482422, 3077.89511719, 3077.80078125, 3076.32236328, 3075.90737305, 3075.41552734, 3074.23193359, 3072.65126953, 3072.25595703, 3073.67148438, 3074.12988281, 3072.24052734, 3069.78862305, 3068.38242188, 3068.2878418, 3068.26259766, 3067.36523438, 3066.35375977, 3065.49633789, 3064.30654297, 3062.81196289, 3061.17758789, 3059.77924805, 3059.14882812, 3059.32856445, 3060.04482422, 3060.61542969, 3060.1449707, 3058.6159668, 3056.63154297, 3054.82905273, 3053.52739258, 3052.54008789, 3051.75473633, 3051.27177734, 3050.94384766, 3050.62529297, 3049.65102539, 3049.13808594, 3046.93251953, 3055.6215332, 3061.47294922, 3055.49760742, 3060.65883789, 3052.26298828, 3057.62397461, 3049.53847656, 3061.0456543, 3052.52456055, 3049.38540039, 3050.13183594, 3052.21186523, 3058.19355469, 3064.17255859, 3062.34677734, 3056.95332031, 3062.87827148, 3059.01484375, 3065.21635742, 3053.95556641, 3062.37724609, 3054.07734375, 3054.16245117, 3054.54047852, 3059.73974609, 3061.6953125, 3052.82382813, 3049.95341797, 3052.97368164, 3054.26967773, 3065.609375, 3055.68706055, 3052.25878906, 3050.01293945, 3048.4203125, 3061.74521484, 3050.7574707, 3046.64492187, 3053.39086914, 3044.52226562, 3044.6394043, 3050.67163086, 3048.26000977, 3043.50639648, 3044.26445312, 3045.20405273, 3041.43388672, 3042.33935547, 3045.86713867, 3042.04799805, 3041.74506836, 3050.41601562, 3067.0137207, 3043.06152344, 3055.70170898, 3047.06953125, 3058.40688477, 3046.86503906, 3049.29355469, 3049.84453125, 3047.90605469, 3056.82802734, 3050.74345703, 3045.91064453, 3047.03125, 3048.07993164, 3053.08134766, 3050.36025391, 3045.22988281, 3047.1387207, 3047.33579102, 3044.23847656, 3045.41738281, 3045.91342773, 3042.19741211, 3040.74956055, 3041.62519531, 3041.14067383, 3041.1706543, 3040.76669922, 3038.92666016, 3037.98969727, 3037.93530273, 3037.04086914, 3036.62592773, 3036.56972656, 3036.0331543, 3034.95751953, 3033.58540039, 3032.72685547, 3032.87802734, 3032.70820313, 3031.86801758, 3031.2965332, 3030.57885742, 3029.36083984, 3028.93842773, 3029.20019531, 3028.62714844, 3027.3512207, 3026.29282227, 3025.66469727, 3025.48334961, 3025.26879883, 3024.40678711, 3023.23227539, 3022.4847168, 3022.18071289, 3021.94384766, 3021.4534668, 3020.62270508, 3019.59580078, 3018.91918945, 3018.70483398, 3018.41337891, 3017.69355469, 3016.79301758, 3016.18588867, 3015.65361328, 3015.10620117, 3014.63911133, 3014.05546875, 3013.43188477, 3012.89375, 3012.33842773, 3011.77763672, 3011.2628418, 3010.72167969, 3010.11245117, 3009.53481445, 3009.03618164, 3008.42832031, 3007.92973633, 3007.50449219, 3006.92275391, 3006.3519043, 3005.77363281, 3005.28598633, 3004.95878906, 3004.42412109, 3003.85717773, 3003.4578125, 3002.97983398, 3002.49780273, 3002.08398438, 3001.65478516, 3001.08959961, 3000.6128418, 3000.27866211, 2999.80644531, 2999.34165039, 2998.86611328, 2998.38422852, 2997.96098633, 2997.53540039, 2997.12050781, 2996.6234375, 2996.17021484, 2995.72578125, 2995.31352539, 2994.89780273, 2994.43916016, 2994.03432617, 2993.60224609, 2993.20566406, 2992.76503906,]
time_our = np.log(np.array(time_our))
#time_our = np.array(time_our)
loss_our = np.array(loss_our)
ind_our = np.argwhere(time_our <= 4000).flatten()


time_farho = [0.15, 0.36734776496887206, 0.7088263988494873, 1.0581182003021241, 1.3978381633758545, 1.7400427341461182, 2.087732744216919, 2.4341171741485597, 2.7710859298706056, 3.1158048629760744, 3.4673654556274416, 3.8189048767089844, 4.191841125488281, 4.566517353057861, 4.923396968841553, 5.284816598892212, 5.633819103240967, 5.992244100570678, 6.3544535636901855, 6.724139738082886, 7.072247266769409, 7.425189065933227, 7.784632635116577, 8.149326133728028, 8.514332151412964, 8.875204229354859, 9.243292903900146, 9.609354209899902, 9.96184515953064, 10.31709303855896, 10.676223134994506, 11.043996286392211, 11.41572380065918, 11.775326013565063, 12.146299409866334, 12.507064771652221, 12.875027418136597, 13.240549755096435, 13.604971265792846, 13.959915971755981, 14.320844221115113, 14.673579359054566, 15.035765075683594, 15.383344841003417, 15.724108743667603, 16.076285362243652, 16.434208869934082, 16.783316087722778, 17.12485065460205, 17.46564702987671, 17.82443962097168, 18.180974674224853, 18.545956897735596, 18.897344207763673, 19.24359107017517, 19.58623399734497, 19.94859070777893, 20.302093839645387, 20.63678035736084, 20.992096853256225, 21.347682762145997, 21.70503649711609, 22.058619928359985, 22.423507976531983, 22.775625371932982, 23.139139127731323, 23.506516551971437, 23.845312356948853, 24.20176763534546, 24.545331478118896, 24.91062989234924, 25.251554441452026, 25.6133309841156, 25.98567509651184, 26.354595184326172, 26.729399967193604, 27.102932691574097, 27.46546835899353, 27.814673709869385, 28.16586298942566, 28.55077199935913, 28.914029836654663, 29.27292046546936, 29.629769039154052, 29.989132738113405, 30.333323526382447, 30.685768413543702, 31.034891653060914, 31.381310415267944, 31.72925992012024, 32.086592292785646, 32.44125232696533, 32.807903099060056, 33.15834245681763, 33.511310291290286, 33.86810717582703, 34.20967626571655, 34.549292373657224, 34.90936045646667, 35.26311883926392, 35.6218466758728, 35.9644202709198, 36.30406002998352, 36.64757404327393, 37.018523931503296, 37.37912392616272, 37.73809790611267, 38.095595073699954, 38.43620309829712, 38.81628155708313, 39.166071557998656, 39.51069526672363, 39.86110916137695, 40.210344552993774, 40.557216024398805, 40.92817387580872, 41.297469758987425, 41.64313998222351, 41.989577770233154, 42.362097597122194, 42.72004008293152, 43.07950391769409, 43.45509705543518, 43.81416974067688, 44.16488156318665, 44.52177100181579, 44.874811458587644, 45.2271547794342, 45.56562008857727, 45.91641297340393, 46.28127388954162, 46.64687285423279, 47.00639796257019, 47.35995268821716, 47.71692056655884, 48.07447810173035, 48.42970581054688, 48.779312705993654, 49.122340297698976, 49.454275131225586, 49.79908928871155, 50.15263271331787, 50.48627896308899, 50.82705726623535, 51.185158491134644, 51.55460658073425, 51.916751289367674, 52.28338179588318, 52.65379033088684, 53.038269472122195, 53.3971025466919, 53.737811183929445, 54.08967247009277, 54.46499662399292, 54.821370363235474, 55.16677303314209, 55.51778273582458, 55.87879047393799, 56.22329978942871, 56.57428555488586, 56.94888863563538, 57.30876111984253, 57.675905752182004, 58.028231906890866, 58.385869121551515, 58.72163796424866, 59.075782251358035, 59.43053340911865, 59.783636474609374, 60.14668135643005, 60.51167607307434, 60.86821804046631, 61.22768630981445, 61.59366250038147, 61.94803166389465, 62.30689058303833, 62.66137609481812, 63.01547484397888, 63.366645669937135, 63.714390182495116, 64.06939301490783, 64.41374239921569, 64.7498980998993, 65.10979561805725, 65.45652179718017, 65.81060800552368, 66.17153911590576, 66.5368929862976, 66.88859767913819, 67.24939856529235, 67.60708351135254, 67.97955226898193, 68.33446054458618, 68.69258165359497, 69.04031772613526, 69.39315428733826, 69.75041427612305, 70.11489391326904, 70.47681546211243, 70.81428246498108, 71.15822005271912, 71.50798115730285, 71.87234406471252, 72.22681217193603, 72.58489317893982, 72.95345301628113, 73.311940574646, 73.66982002258301, 74.02340564727783, 74.36988558769227, 74.72167472839355, 75.08238582611084, 75.43668427467347, 75.78814392089843, 76.14121112823486, 76.50298438072204, 76.85014758110046, 77.20921893119812, 77.57598028182983, 77.9334376335144, 78.3104956626892, 78.67672128677368, 79.04216084480285, 79.37652540206909, 79.72201824188232, 80.07224493026733, 80.42846684455871, 80.77394380569459, 81.14726943969727, 81.50554189682006, 81.86798892021179, 82.20686693191529, 82.5541501045227, 82.90028762817383, 83.24421715736389, 83.59328856468201, 83.94289412498475, 84.29698958396912, 84.6636917591095, 85.014821767807, 85.36659393310546, 85.70948491096496, 86.06877951622009, 86.45044460296631, 86.80613899230957, 87.16817002296447, 87.53820672035218, 87.898766374588, 88.28154668807983, 88.63952436447144, 89.00083065032959, 89.36077361106872, 89.7038897037506, 90.04582438468933, 90.41015686988831, 90.75612034797669, 91.09202971458436, 91.44175825119018, 91.7939181804657, 92.16657018661499, 92.53093223571777, 92.90558490753173, 93.28252968788146, 93.64065303802491, 93.99569745063782, 94.34527812004089, 94.71337642669678, 95.06792688369751, 95.42388710975646, 95.76262845993043, 96.10555901527405, 96.44847893714905, 96.78438429832458, 97.13590564727784, 97.48861784934998, 97.82889590263366, 98.17359042167664, 98.5041877269745, 98.85290713310242, 99.20320262908936, 99.56619987487792, 99.93385453224182, 100.31269073486328, 100.67495293617249, 101.03568558692932, 101.41174592971802, 101.77417645454406, 102.12196230888367, 102.49065794944764, 102.84947128295899, 103.19203023910522, 103.54947652816773, 103.89952116012573, 104.25111417770385, 104.61281304359436, 104.96470437049865, 105.31642518043517, 105.6907991886139, 106.05515093803406, 106.42595624923706, 106.77895617485046, 107.11285781860352, 107.46291851997375, 107.81402640342712, 108.15740394592285, 108.52337665557862, 108.88568315505981, 109.25690741539002, 109.61026482582092, 109.98776021003724, 110.33671164512634, 110.68314905166626, 111.03509721755981, 111.40834226608277, 111.7552421092987, 112.10305457115173, 112.45985651016235, 112.8221351146698, 113.18922209739685, 113.53368501663208, 113.88421149253845, 114.24023752212524, 114.58057980537414, 114.93756980895996, 115.29777092933655, 115.63683691024781, 115.99050211906433, 116.35581274032593, 116.6995231628418, 117.05853986740112, 117.40839729309081, 117.75971636772155, 118.11349396705627, 118.48260712623596, 118.83673014640809, 119.17300944328308, 119.52109813690186, 119.86868953704834, 120.20548567771911, 120.55533032417297, 120.90947132110595, 121.25508685112, 121.58646259307861, 121.9433069229126, 122.3014187335968, 122.65412468910218, 123.03122854232788, 123.40355243682862, 123.7704357624054, 124.12237300872803, 124.5055299282074, 124.86785631179809, 125.23167705535889, 125.59177479743957, 125.94657173156739, 126.28926634788513, 126.64063563346863, 126.9918749332428, 127.32514934539795, 127.6645664215088, 128.01484351158143, 128.35941996574402, 128.73847098350524, 129.0894917011261, 129.44095468521118, 129.80150904655457, 130.13947467803956, 130.4869565963745, 130.83399457931517, 131.1848906517029, 131.54963445663452, 131.91447825431823, 132.27439770698547, 132.63660349845887, 132.98853807449342, 133.33424167633058, 133.66963481903076, 134.01382966041564, 134.39557700157167, 134.7380425453186, 135.08918056488037, 135.44029936790466, 135.79421305656433, 136.14418544769288, 136.49920935630797, 136.84047856330872, 137.20066838264466, 137.5545135974884, 137.93755774497987, 138.29211621284486, 138.64262804985046, 138.99169783592225, 139.3414439201355, 139.68491806983948, 140.02430057525635, 140.37529792785645, 140.7269564628601, 141.07478675842285, 141.41738386154174, 141.763623046875, 142.1023880958557, 142.47500762939453, 142.83847489356995, 143.20859589576722, 143.57051029205323, 143.91886773109437, 144.28485345840454, 144.63964223861694, 145.00162515640258, 145.34577717781067, 145.6970322608948, 146.06568179130554, 146.40906205177308, 146.78126502037048, 147.15004057884215, 147.50744891166687, 147.8600998878479, 148.21914558410646, 148.57263164520265, 148.92711853981018, 149.29906692504883, 149.65879135131837, 150.00912461280822, 150.3565954208374, 150.70692400932313, 151.0580159664154, 151.40281200408936, 151.7378713130951, 152.08915634155272, 152.43094868659972, 152.77196168899536, 153.11522965431215, 153.46020698547363, 153.80559992790222, 154.16419467926025, 154.51395964622498, 154.87432594299315, 155.22278995513915, 155.59245142936706, 155.97725439071655, 156.34344024658202, 156.70085301399232, 157.07334365844727, 157.4338500022888, 157.78623113632202, 158.14694318771362, 158.52616653442382, 158.8793550491333, 159.24128890037537, 159.5800971031189, 159.92572097778321, 160.2717574596405, 160.61802697181702, 160.96628198623657, 161.30338826179505, 161.66318707466127, 162.0169846057892, 162.37111973762512, 162.74034237861633, 163.0992133617401, 163.44402470588685, 163.79784488677979, 164.17277183532715, 164.51585426330567, 164.86749224662782, 165.20531702041626, 165.54063467979432, 165.8960753440857, 166.23810992240905, 166.57845253944396, 166.91636242866517, 167.26346201896666, 167.6122953414917, 167.95899295806885, 168.3195222377777, 168.6742844581604, 169.04257106781006, 169.3880359172821, 169.73265771865846, 170.0798813819885, 170.4295732975006, 170.77959728240967, 171.13001465797424, 171.49785642623903, 171.83645205497743, 172.1883940219879, 172.56090683937072, 172.92090740203858, 173.28729209899902, 173.6460129737854, 173.9980996131897, 174.3566770553589, 174.72545380592345, 175.09693837165833, 175.459397315979, 175.82611513137817, 176.184467792511, 176.55687880516052, 176.90677766799928, 177.2557393550873, 3356]
loss_farho = [20000, 18656.261328125, 17544.162890625, 16528.926953125, 15603.8412109375, 14749.151953125, 13955.9841796875, 13223.524609375, 12547.6134765625, 11922.637890625, 11344.387890625, 10809.8173828125, 10316.3916015625, 9861.746875, 9443.55390625, 9059.482421875, 8707.222265625, 8384.50703125, 8089.1212890625, 7818.86806640625, 7571.5703125, 7345.08544921875, 7137.337109375, 6946.36767578125, 6770.3529296875, 6607.64990234375, 6456.78369140625, 6316.46591796875, 6185.58173828125, 6063.173828125, 5948.43203125, 5840.6705078125, 5739.30634765625, 5643.831640625, 5553.8037109375, 5468.8154296875, 5388.49580078125, 5312.500390625, 5240.50947265625, 5172.2365234375, 5107.4224609375, 5045.8525390625, 4987.3435546875, 4931.7421875, 4878.91953125, 4828.7626953125, 4781.1607421875, 4735.99755859375, 4693.14521484375, 4652.4556640625, 4613.76552734375, 4576.89814453125, 4541.66982421875, 4507.90771484375, 4475.4568359375, 4444.18203125, 4413.98193359375, 4384.776171875, 4356.50830078125, 4329.1357421875, 4302.62685546875, 4276.95107421875, 4252.07978515625, 4227.98203125, 4204.6240234375, 4181.96845703125, 4159.97841796875, 4138.61474609375, 4117.83837890625, 4097.615234375, 4077.91162109375, 4058.69921875, 4039.95390625, 4021.6564453125, 4003.792529296875, 3986.35126953125, 3969.322607421875, 3952.70380859375, 3936.488427734375, 3920.67490234375, 3905.258251953125, 3890.233935546875, 3875.597509765625, 3861.33984375, 3847.450341796875, 3833.91591796875, 3820.71982421875, 3807.844287109375, 3795.269091796875, 3782.975, 3770.944140625, 3759.159912109375, 3747.610302734375, 3736.28662109375, 3725.181884765625, 3714.291552734375, 3703.612109375, 3693.13828125, 3682.865234375, 3672.789794921875, 3662.902783203125, 3653.1994140625, 3643.673291015625, 3634.319873046875, 3625.131494140625, 3616.10224609375, 3607.229248046875, 3598.5060546875, 3589.9271484375, 3581.489697265625, 3573.190966796875, 3565.028515625, 3557.000146484375, 3549.107666015625, 3541.349462890625, 3533.7263671875, 3526.235400390625, 3518.875048828125, 3511.641650390625, 3504.53037109375, 3497.537255859375, 3490.65791015625, 3483.891357421875, 3477.2369140625, 3470.69345703125, 3464.2638671875, 3457.946826171875, 3451.743408203125, 3445.65302734375, 3439.6724609375, 3433.79931640625, 3428.02880859375, 3422.357080078125, 3416.77939453125, 3411.291357421875, 3405.888232421875, 3400.5677734375, 3395.3255859375, 3390.1611328125, 3385.072705078125, 3380.060400390625, 3375.124658203125, 3370.264794921875, 3365.48193359375, 3360.7765625, 3356.1490234375, 3351.598193359375, 3347.1224609375, 3342.722021484375, 3338.394189453125, 3334.13564453125, 3329.94521484375, 3325.8197265625, 3321.757373046875, 3317.754736328125, 3313.810009765625, 3309.922265625, 3306.08828125, 3302.305224609375, 3298.5759765625, 3294.89443359375, 3291.263134765625, 3287.680322265625, 3284.14404296875, 3280.655078125, 3277.21123046875, 3273.8140625, 3270.461181640625, 3267.153173828125, 3263.889990234375, 3260.670361328125, 3257.49404296875, 3254.3611328125, 3251.270556640625, 3248.222265625, 3245.216455078125, 3242.2505859375, 3239.324951171875, 3236.43984375, 3233.593603515625, 3230.78525390625, 3228.0154296875, 3225.282373046875, 3222.58623046875, 3219.92626953125, 3217.302197265625, 3214.713134765625, 3212.15810546875, 3209.637646484375, 3207.14951171875, 3204.69453125, 3202.27119140625, 3199.878955078125, 3197.5171875, 3195.18525390625, 3192.88232421875, 3190.60791015625, 3188.36259765625, 3186.145849609375, 3183.956884765625, 3181.795849609375, 3179.662353515625, 3177.55625, 3175.4767578125, 3173.424169921875, 3171.397509765625, 3169.396728515625, 3167.422509765625, 3165.47236328125, 3163.54697265625, 3161.64580078125, 3159.769091796875, 3157.9162109375, 3156.085986328125, 3154.279345703125, 3152.494677734375, 3150.733740234375, 3148.99287109375, 3147.274609375, 3145.577197265625, 3143.9017578125, 3142.246630859375, 3140.611767578125, 3138.997900390625, 3137.403564453125, 3135.82919921875, 3134.27451171875, 3132.738818359375, 3131.22294921875, 3129.72646484375, 3128.2474609375, 3126.788037109375, 3125.34716796875, 3123.923291015625, 3122.51787109375, 3121.13017578125, 3119.7599609375, 3118.406494140625, 3117.070556640625, 3115.7509765625, 3114.447900390625, 3113.16103515625, 3111.89072265625, 3110.63623046875, 3109.3982421875, 3108.17431640625, 3106.96640625, 3105.773681640625, 3104.59619140625, 3103.43369140625, 3102.285791015625, 3101.15263671875, 3100.033837890625, 3098.928662109375, 3097.83837890625, 3096.762060546875, 3095.698291015625, 3094.64990234375, 3093.61318359375, 3092.590771484375, 3091.581396484375, 3090.58564453125, 3089.6015625, 3088.630517578125, 3087.673486328125, 3086.728125, 3085.794970703125, 3084.873388671875, 3083.965283203125, 3083.0677734375, 3082.18291015625, 3081.309716796875, 3080.448681640625, 3079.5974609375, 3078.759375, 3077.93154296875, 3077.11474609375, 3076.309716796875, 3075.514892578125, 3074.73125, 3073.958056640625, 3073.1953125, 3072.4431640625, 3071.70185546875, 3070.96943359375, 3070.248779296875, 3069.53740234375, 3068.836474609375, 3068.144482421875, 3067.462841796875, 3066.791162109375, 3066.129052734375, 3065.476025390625, 3064.83291015625, 3064.198779296875, 3063.5740234375, 3062.9578125, 3062.351171875, 3061.752978515625, 3061.164404296875, 3060.584375, 3060.0125, 3059.45, 3058.89541015625, 3058.349951171875, 3057.811962890625, 3057.283349609375, 3056.7625, 3056.250146484375, 3055.7453125, 3055.2486328125, 3054.760546875, 3054.27900390625, 3053.806103515625, 3053.341015625, 3052.883056640625, 3052.43330078125, 3051.990771484375, 3051.55625, 3051.12841796875, 3050.708447265625, 3050.295361328125, 3049.889892578125, 3049.491259765625, 3049.0990234375, 3048.71494140625, 3048.337255859375, 3047.966748046875, 3047.602734375, 3047.24599609375, 3046.895068359375, 3046.552001953125, 3046.214794921875, 3045.884228515625, 3045.559814453125, 3045.24248046875, 3044.931689453125, 3044.626611328125, 3044.328564453125, 3044.035400390625, 3043.74931640625, 3043.469921875, 3043.195556640625, 3042.9279296875, 3042.666162109375, 3042.40986328125, 3042.160009765625, 3041.91591796875, 3041.677099609375, 3041.444677734375, 3041.217724609375, 3040.9966796875, 3040.7810546875, 3040.570849609375, 3040.366015625, 3040.1669921875, 3039.9734375, 3039.78505859375, 3039.6021484375, 3039.4248046875, 3039.251513671875, 3039.084619140625, 3038.923046875, 3038.76611328125, 3038.61396484375, 3038.46806640625, 3038.326513671875, 3038.18984375, 3038.05830078125, 3037.93134765625, 3037.8091796875, 3037.692626953125, 3037.580810546875, 3037.47294921875, 3037.37041015625, 3037.27236328125, 3037.17939453125, 3037.090625, 3037.006640625, 3036.926904296875, 3036.851904296875, 3036.78125, 3036.715087890625, 3036.65390625, 3036.596484375, 3036.5435546875, 3036.49560546875, 3036.450927734375, 3036.410986328125, 3036.375341796875, 3036.34384765625, 3036.31669921875, 3036.2935546875, 3036.274267578125, 3036.259326171875, 3036.249560546875, 3036.24169921875, 3036.239208984375, 3036.241064453125, 3036.2458984375, 3036.254833984375, 3036.26806640625, 3036.28466796875, 3036.305615234375, 3036.32958984375, 3036.3578125, 3036.390283203125, 3036.42548828125, 3036.465283203125, 3036.508447265625, 3036.555419921875, 3036.6056640625, 3036.660546875, 3036.718359375, 3036.77890625, 3036.843408203125, 3036.912060546875, 3036.983642578125, 3037.05966796875, 3037.13837890625, 3037.22021484375, 3037.305908203125, 3037.394775390625, 3037.487060546875, 3037.582373046875, 3037.681201171875, 3037.783740234375, 3037.889501953125, 3037.99814453125, 3038.11025390625, 3038.225048828125, 3038.34345703125, 3038.465185546875, 3038.589892578125, 3038.71728515625, 3038.848291015625, 3038.9826171875, 3039.119677734375, 3039.25947265625, 3039.40322265625, 3039.5494140625, 3039.697900390625, 3039.850146484375, 3040.005517578125, 3040.16318359375, 3040.32421875, 3040.4884765625, 3040.654931640625, 3040.824365234375, 3040.99658203125, 3041.17158203125, 3041.34990234375, 3041.53046875, 3041.714794921875, 3041.90166015625, 3042.0900390625, 3042.281982421875, 3042.47587890625, 3042.674560546875, 3042.873828125, 3043.07626953125, 3043.28193359375, 3043.489697265625, 3043.70009765625, 3043.9130859375, 3044.128466796875, 3044.347314453125, 3044.56826171875, 3044.79130859375, 3045.01689453125, 3045.24541015625, 3045.47666015625, 3045.709912109375, 3045.945263671875, 3046.18369140625, 3046.42412109375, 3046.667919921875, 3046.91318359375, 3047.160986328125, 3047.411181640625, 3047.664599609375, 3047.9193359375, 3048.17646484375, 3048.436767578125, 3048.698486328125, 3048.962353515625, 3049.2291015625, 3049.49794921875, 3049.769287109375, 3050.04228515625, 3050.318408203125, 3050.596533203125, 3050.87685546875, 3051.159130859375, 3051.44296875, 3051.730078125, 3052.018505859375, 3052.30927734375, 3052.603271484375, 3052.898095703125, 3053.1958984375,]
time_farho = np.log(np.array(time_farho))
#time_farho = np.array(time_farho)
loss_farho = np.array(loss_farho)
ind_farho = np.argwhere(time_farho <= 4000).flatten()


time_hoag = [0.15, 225.002986479, 3356.91766229]
loss_hoag = [20000, 2840.04976434, 2846.03958849]
time_hoag = np.log(np.array(time_hoag))
#time_hoag = np.array(time_hoag)
loss_hoag = np.array(loss_hoag)
ind_hoag = np.argwhere(time_hoag <= 4000).flatten()


plt.figure(1)
plt.title("Test Loss Vs Time on MNIST (28X28) dataset")  
plt.xlabel("Time (seconds)")
plt.ylabel("Loss on Test Set")
#plt.plot(time_our[ind_our], loss_our[ind_our], 'bo', time_our[ind_our], loss_our[ind_our], 'b', label="Our")
#plt.plot(time_farho[ind_farho], loss_farho[ind_farho], 'co', time_farho[ind_farho], loss_farho[ind_farho], 'c', label="FARHO")
#plt.plot(time_our_gpu[ind_our_gpu], loss_our_gpu[ind_our_gpu], 'mo', time_our_gpu[ind_our_gpu], loss_our_gpu[ind_our_gpu], 'm', label="Our GPU")
#plt.plot(time_farho_gpu[ind_farho_gpu], loss_farho_gpu[ind_farho_gpu], 'go', time_farho_gpu[ind_farho_gpu], loss_farho_gpu[ind_farho_gpu], 'g', label="FARHO GPU")
#plt.plot(time_hoag[ind_hoag], loss_hoag[ind_hoag], 'ro', time_hoag[ind_hoag], loss_hoag[ind_hoag], 'r', label="HOAG")
plt.plot(time_our[ind_our], loss_our[ind_our], 'b', label="Our")
plt.plot(time_farho[ind_farho], loss_farho[ind_farho], 'g', label="FARHO")
plt.plot(time_hoag[ind_hoag], loss_hoag[ind_hoag], 'r', label="HOAG")
#plt.plot(time_gs[ind_gs], loss_gs[ind_gs], 'c', label="GRID SEARCH")
#plt.plot(time_opt[ind_opt], loss_opt[ind_opt], 'm', label="Optimal with Train + Val Set")
#plt.axis([0, 5, 2500, 4000])
#plt.yticks(np.arange(0, 3600, 200))
#plt.xticks(np.arange(0.0, 3600, 200))
plt.legend(loc="best")
plt.show()
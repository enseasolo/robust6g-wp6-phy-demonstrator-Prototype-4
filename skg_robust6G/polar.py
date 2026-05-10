# -*- coding: utf-8 -*-
"""
===============
Polar Codes
===============
"""
from __future__ import annotations
from sympy.combinatorics import GrayCode
import numpy as np
import math
from copy import copy
from typing import Optional, TYPE_CHECKING
from scipy import signal, linalg
import cmath
import operator
import matplotlib.pyplot as plt


__author__ = "Amitha Mayya"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Amitha Mayya, Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Amitha Mayya"
__email__ = "amitha.mayya@barkhauseninstitut.org"
__status__ = "Prototype"

class PolarCodes:
    """A class to generate syndrome and syndrome decoder for Polar codes with Slepian-Wolf implementation"""
    def __init__(self, 
                info_recon: InformationReconciliation = None,
                error_prob: Optional[float] = 0.01,
                *args,
                **kwargs) ->None:

        self.info_recon = info_recon
        self.error_prob = error_prob
        """
        Args:
        info_recon : The handle to the information reconciliation class which contains the codewords, coderate and type of the code   
        error_prob : Probability of error
        """
    def SW_Error_Correction(self) -> Tuple[bool, np.ndarray, np.ndarray, int]:
        """Implements Error Correction using Slepian Wolf algorithm and Polar Codes for the channel codes
        Returns:
            
            recon_code_one (np.ndarray): Reconciliated code after Slepian-Wolf for code one
            recon_code_two (np.ndarray): Reconciliated code after Slepian-Wolf for code two
            
        """
        #Define the parameters (general for SC and CRC decoder)
        ######################################################################
        # self.Q = self.BitChannel_order().flatten()+1
        self.SNR_dB = self.info_recon.SNR_dB
        self.Q = self.BitChannel_order(self.SNR_dB).flatten()+1
        ######################################################################
        # self.Q = self.BitChannel_order().flatten()+1
        self.N = self.info_recon.codeword_length
        self.n = np.log2(self.N)
        #Check for power of 2
        if (not(math.log(self.N, 2).is_integer())):
            raise ValueError("Wrong input: Length of Bobdata and polar_code ",self.N ," is not a power of 2!")
        else:
            self.n = int(self.n)  #Typecast into integer
        entropy = -self.error_prob*np.log2(self.error_prob) - (1-self.error_prob)*np.log2((1-self.error_prob))
        self.Q1 = self.Q[self.Q <= self.N] #reliability sequence for N
        
        
        if self.info_recon.block_error_code == 'Polar_SC':
            #Additional parameter for SC decoder
            self.K = math.floor(self.info_recon.code_rate * self.N)
            self.F = self.Q1[:(self.N-self.K)] #Frozen positions: Q1(1:N-K), Message positions: Q1(N-K+1:end)
            #Generate the Syndrome for the first code
            msg_cap, frozen_values = self.SW_Encoder (self.info_recon.code_one)
            #Decode the second code
            recon_code_one, recon_code_two = self.SC_Decoder(msg_cap, frozen_values,self.error_prob,self.info_recon.code_two)
            
        
        if self.info_recon.block_error_code == 'Polar_CRC':
            #Additional parameters for CRC decoder
            self.A = math.floor(self.info_recon.code_rate * self.N)
            self.crcL = 11
            crcg =np.array([1, 1, 1 ,0, 0, 0, 1 ,0 ,0, 0, 0 ,1])  #CRC polynomial %fliplr([1 1 1 0 1 0 1 0 1])
            self.crcg = crcg[::-1]
            self.K = self.A + self.crcL #CRC length = crcL
            self.F = self.Q1[:(self.N-self.K)] #Frozen positions: Q1(1:N-K), Message positions: Q1(N-K+1:end)
            self.rmax = 10000 #max received value
            self.maxqr = 10000 #max integer received value
            #Generate the Syndrome for the first code
            msg_cap, frozen_values = self.SW_Encoder (self.info_recon.code_one)
            #Decode the second code
            recon_code_one,recon_code_two,cant_reconcile = self.CRC_Decoder(msg_cap, frozen_values,self.error_prob,self.info_recon.code_two)
    
        return recon_code_one, recon_code_two


    # def BitChannel_order(self) -> np.ndarray:
    #     """Bit probability array for the channel
    #         Returns :
    #     np.ndarray : Bit probability in the given channel
    #     """
    #     bit_channel = np.load("/users/atsupass55/Authentication_reconcil based/Preprocessing/bit_channel.npz", allow_pickle=True)
    #     Q = bit_channel["Q"] 

    #     print("Q ", Q)

    #     # bit_channel = np.savez("/users/atsupass55/Authentication_reconcil based/Preprocessing/bit_channel.npz",  allow_pickle=True)
    #     # Q = bit_channel["Q"] 
    #     return Q
    
    ############################################################################################################################
    def BitChannel_order(self, SNR_dB) -> np.ndarray:
        """Bit probability array for the channel
            Returns :
        np.ndarray : Bit probability in the given channel
        """
        SNR_dB = self.SNR_dB

        bit_channel = np.load("bit_channel.npz", allow_pickle=True)
        Q_snr = bit_channel["Q_snr"].item() 

        Q = Q_snr[int(SNR_dB)]

        # print("Q ", Q)
        return Q
    ###############################################################################################################################

    
    def BitChannel_order__(self) -> np.ndarray:
        """Bit probability array for the channel
            Returns :
        np.ndarray : Bit probability in the given channel
        """
        Q = [0,1,2,4,8,16,32,3,5,64,9,6,17,10,18,128,12,33,65,20,256,34,24,36,7,129,66,512,11,40,68,130,
            19,13,48,14,72,257,21,132,35,258,26,513,80,37,25,22,136,260,264,38,514,96,67,41,144,28,69,42,516,49,74,272,
            160,520,288,528,192,544,70,44,131,81,50,73,15,320,133,52,23,134,384,76,137,82,56,27,97,39,259,84,138,145,261,29,
            43,98,515,88,140,30,146,71,262,265,161,576,45,100,640,51,148,46,75,266,273,517,104,162,53,193,152,77,164,768,268,274,
            518,54,83,57,521,112,135,78,289,194,85,276,522,58,168,139,99,86,60,280,89,290,529,524,196,141,101,147,176,142,530,321,
            31,200,90,545,292,322,532,263,149,102,105,304,296,163,92,47,267,385,546,324,208,386,150,153,165,106,55,328,536,577,548,113,
            154,79,269,108,578,224,166,519,552,195,270,641,523,275,580,291,59,169,560,114,277,156,87,197,116,170,61,531,525,642,281,278,
            526,177,293,388,91,584,769,198,172,120,201,336,62,282,143,103,178,294,93,644,202,592,323,392,297,770,107,180,151,209,284,648,
            94,204,298,400,608,352,325,533,155,210,305,547,300,109,184,534,537,115,167,225,326,306,772,157,656,329,110,117,212,171,776,330,
            226,549,538,387,308,216,416,271,279,158,337,550,672,118,332,579,540,389,173,121,553,199,784,179,228,338,312,704,390,174,554,581,
            393,283,122,448,353,561,203,63,340,394,527,582,556,181,295,285,232,124,205,182,643,562,286,585,299,354,211,401,185,396,344,586,
            645,593,535,240,206,95,327,564,800,402,356,307,301,417,213,568,832,588,186,646,404,227,896,594,418,302,649,771,360,539,111,331,
            214,309,188,449,217,408,609,596,551,650,229,159,420,310,541,773,610,657,333,119,600,339,218,368,652,230,391,313,450,542,334,233,
            555,774,175,123,658,612,341,777,220,314,424,395,673,583,355,287,183,234,125,557,660,616,342,316,241,778,563,345,452,397,403,207,
            674,558,785,432,357,187,236,664,624,587,780,705,126,242,565,398,346,456,358,405,303,569,244,595,189,566,676,361,706,589,215,786,
            647,348,419,406,464,680,801,362,590,409,570,788,597,572,219,311,708,598,601,651,421,792,802,611,602,410,231,688,653,248,369,190,
            364,654,659,335,480,315,221,370,613,422,425,451,614,543,235,412,343,372,775,317,222,426,453,237,559,833,804,712,834,661,808,779,
            617,604,433,720,816,836,347,897,243,662,454,318,675,618,898,781,376,428,665,736,567,840,625,238,359,457,399,787,591,678,434,677,
            349,245,458,666,620,363,127,191,782,407,436,626,571,465,681,246,707,350,599,668,790,460,249,682,573,411,803,789,709,365,440,628,
            689,374,423,466,793,250,371,481,574,413,603,366,468,655,900,805,615,684,710,429,794,252,373,605,848,690,713,632,482,806,427,904,
            414,223,663,692,835,619,472,455,796,809,714,721,837,716,864,810,606,912,722,696,377,435,817,319,621,812,484,430,838,667,488,239,
            378,459,622,627,437,380,818,461,496,669,679,724,841,629,351,467,438,737,251,462,442,441,469,247,683,842,738,899,670,783,849,820,
            728,928,791,367,901,630,685,844,633,711,253,691,824,902,686,740,850,375,444,470,483,415,485,905,795,473,634,744,852,960,865,693,
            797,906,715,807,474,636,694,254,717,575,913,798,811,379,697,431,607,489,866,723,486,908,718,813,476,856,839,725,698,914,752,868,
            819,814,439,929,490,623,671,739,916,463,843,381,497,930,821,726,961,872,492,631,729,700,443,741,845,920,382,822,851,730,498,880,
            742,445,471,635,932,687,903,825,500,846,745,826,732,446,962,936,475,853,867,637,907,487,695,746,828,753,854,857,504,799,255,964,
            909,719,477,915,638,748,944,869,491,699,754,858,478,968,383,910,815,976,870,917,727,493,873,701,931,756,860,499,731,823,922,874,
            918,502,933,743,760,881,494,702,921,501,876,847,992,447,733,827,934,882,937,963,747,505,855,924,734,829,965,938,884,506,749,945,
            966,755,859,940,830,911,871,639,888,479,946,750,969,508,861,757,970,919,875,862,758,948,977,923,972,761,877,952,495,703,935,978,
            883,762,503,925,878,735,993,885,939,994,980,926,764,941,967,886,831,947,507,889,984,751,942,996,971,890,509,949,973,1000,892,950,
            863,759,1008,510,979,953,763,974,954,879,981,982,927,995,765,956,887,985,997,986,943,891,998,766,511,988,1001,951,1002,893,975,894,
            1009,955,1004,1010,957,983,958,987,1012,999,1016,767,989,1003,990,1005,959,1011,1013,895,1006,1014,1017,1018,991,1020,1007,1015,1019,1021,1022,1023]
    
        return np.array(Q, dtype=int)
    
    
    def SW_Encoder (self,polar_code,er_prob = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Implements the Slepian Wolf Encoder to generate the syndrome
            Args:
                polar_code (np.ndarray) : The polar code to generate the syndrome
            Returns:
                msg_cap (np.ndarray) : Messgae positions of the polar code 
                frozen_values (np.ndarray): Frozen positions of the polar code (syndrome)
        """
        n = self.n
        N = self.N
        Q1 = self.Q1
        K = self.K
        L = np.zeros((n+1,N)) #beliefs
        ucap = np.zeros((n+1,N)) #decisions
        ns = np.zeros((2*N-1)); #node state vector
        er_prob_ratio = (1-er_prob)/er_prob
        L[0,:] = (1-2*polar_code)*er_prob_ratio; #belief of root
        
        node = 0
        depth = 0 #start at the root
        done = 0 #decoder has finished

        while (done == 0):   #traverse till all bits are decoded
            #leaf of the root
            if ( depth == n):
                if ( L[(n+1)-1,(node+1)-1] >= 0):
                    ucap[(n+1)-1,(node+1)-1] = 0
                else:
                    ucap[(n+1)-1,(node+1)-1] = 1
            
                if(node == N-1):
                    done = 1
                else:
                    node = math.floor(node/2) 
                    depth = depth - 1
            
            else:
                #nonleaf
                npos = (2 ** depth-1 + node + 1)-1   #position of node in node state vector
                if (ns[npos] == 0): #step L and go to left child
                    temp = 2 ** (n-depth)
                    Ln = L[(depth+1)-1,(temp*node+1)-1:(temp*(node+1))] #incoming beliefs
                    a,b = np.split(Ln,2) #split beliefs into 2
                    node = node *2    #next node: left child
                    depth = depth +  1
                    temp = int(temp / 2 )   #incoming belief length for left child
                    L[(depth+1)-1,(temp*node+1)-1:temp*(node+1)] = self.f(a,b);   #minsum and storage
                    ns[npos] = 1

                else:
                    if (ns[npos] == 1) : #step R and go to right child
                        temp = 2 ** (n-depth)
                        Ln = L[(depth+1)-1,(temp*node+1)-1:temp*(node+1)] #incoming beliefs
                        a,b = np.split(Ln,2) #split beliefs into 2
                        lnode = 2*node       #left child
                        ldepth = depth + 1
                        ltemp = int(temp/2)
                        ucapn = ucap[(ldepth+1)-1,(ltemp*lnode+1)-1:ltemp*(lnode+1)]  #incoming decisions from left child
                        node = node *2 + 1  #next node: right child
                        depth = depth + 1   #incoming belief length for right child
                        temp = int(temp/2)
                        test_store = self.g(a,b,ucapn)
                        L[(depth+1)-1,((temp*node+1)-1):temp*(node+1)] = self.g(a,b,ucapn)  #g and storage
                        ns[npos] = 2
                    
                    else: #step U and go to parent
                        temp = 2 ** (n-depth)
                        lnode = 2*node
                        rnode = 2*node + 1
                        cdepth = depth + 1  #left and right child
                        ctemp = int(temp/2)
                        ucapl = ucap[(cdepth+1)-1,(ctemp*lnode+1)-1:ctemp*(lnode+1)] #incoming decisions from left child
                        ucapr = ucap[(cdepth+1)-1,(ctemp*rnode+1)-1:ctemp*(rnode+1)] #incoming decisions from right child
                        ucap[(depth+1)-1,(temp*node+1)-1:temp*(node+1)] = np.concatenate([(ucapl+ucapr) %2, ucapr]) #combine
                        node = math.floor(node/2)
                        depth = depth - 1

        msg_vector = Q1[N-K:]
        frozen_vector = Q1[:N-K] 
        msg_cap = [ucap[n,msg_vector[i]-1] for i in range(len(msg_vector))]
        frozen_values = [ucap[n, frozen_vector[i]-1] for i in range(len(frozen_vector))]

        return msg_cap, frozen_values

    def SC_Decoder(self,msg_cap,frozen_values,error_prob,polar_code) -> Tuple[bool, np.ndarray, np.ndarray, int]:
        """Implements Successive Cancellation Decoder for Polar Codes
        Args:
            msg_cap (np.ndarray): Message part of the syndrome, must be moved outside for online reconciliation
            frozen_values (np.ndarray) : Frozen values of the syndrome used to dcode the polar code
            error_prob (float): Error Probability
            polar_code (np.ndarray): Polar code to be decoded

        Returns:
            Success_recon (bool) : Flag indicating reconciliation succeeded or not
            recon_code_one (np.ndarray) : First reconciliated code
            recon_code_two (np.ndarray) : Second reconciliated code
            Nerrs (int) : Number of errors after reconciliation
        
        """
        
        n = self.n
        N = self.N
        Q1 = self.Q1
        K = self.K
        F = self.F
        u = np.zeros(N)
        frozen_indices = np.array(Q1[:N-K])-1
        msg_indices = np.array(Q1[N-K:])-1
        u[frozen_indices] = frozen_values
        u[msg_indices] = msg_cap
        u_source = u
        error_prob_ratio = (1-error_prob) / error_prob
        r = (1 - 2*polar_code) * error_prob_ratio #belief of root; AWGN channel I

        #SC Decoder
        L = np.zeros((n+1,N))   #beliefs
        ucap = np.zeros((n+1,N)) #decisions
        ns = np.zeros((2*N-1)) #node state vector

        L[0,:] = r #beleif of root
        node = 0    #start at root
        depth = 0
        done = 0   #decoder has finished or not


        while(done == 0): #traverse till all bits are decoded
            if ( depth == n): #leaf or not
                if((node+1) in F):  #is node frozen
                    ucap[n,node] = u_source[node]
                else:
                    if ( L[n,node] >= 0):
                        ucap[n,node] = 0

                    else:
                        ucap[n,node] = 1
                if(node == (N-1)):
                    done = 1
                else:
                    node = math.floor(node /2)
                    depth = depth -1

            else:  #nonleaf
                npos = (2 ** depth-1 + node + 1)-1 #Position of node in node state vector
                if(ns[npos] == 0):  #Step L and go to the left child
                    temp = 2 **(n-depth)
                    Ln = L[(depth+1)-1,(temp*node+1)-1:(temp*(node+1))] #incoming beliefs
                    a,b = np.split(Ln,2) #split beliefs into 2
                    node = node *2    #next node: left child
                    depth = depth +  1
                    temp = int(temp / 2 )   #incoming belief length for left child
                    L[(depth+1)-1,(temp*node+1)-1:temp*(node+1)] = self.f(a,b);   #minsum and storage
                    ns[npos] = 1
                else:
                    if(ns[npos] == 1):  #step R and go to right child
                        temp = 2 ** (n-depth)
                        Ln = L[(depth+1)-1,(temp*node+1)-1:temp*(node+1)] #incoming beliefs
                        a,b = np.split(Ln,2) #split beliefs into 2
                        #a = Ln[0:temp/2]   #split beliefs into 2
                        #b = Ln[temp/2+1:end]
                        lnode = 2*node       #left child
                        ldepth = depth + 1
                        ltemp = int(temp/2)
                        ucapn = ucap[(ldepth+1)-1,(ltemp*lnode+1)-1:ltemp*(lnode+1)]  #incoming decisions from left child
                        node = node *2 + 1  #next node: right child
                        depth = depth + 1   #incoming belief length for right child
                        temp = int(temp/2)
                        test_store = self.g(a,b,ucapn)
                        L[(depth+1)-1,((temp*node+1)-1):temp*(node+1)] = self.g(a,b,ucapn)  #g and storage
                        ns[npos] = 2

                    else: 
                        #step U and go to parent
                        temp = 2 ** (n-depth)
                        lnode = 2*node
                        rnode = 2*node + 1
                        cdepth = depth + 1  #left and right child
                        ctemp = int(temp/2)
                        ucapl = ucap[(cdepth+1)-1,(ctemp*lnode+1)-1:ctemp*(lnode+1)] #incoming decisions from left child
                        ucapr = ucap[(cdepth+1)-1,(ctemp*rnode+1)-1:ctemp*(rnode+1)] #incoming decisions from right child
                        ucap[(depth+1)-1,(temp*node+1)-1:temp*(node+1)] = np.concatenate([(ucapl+ucapr) %2, ucapr]) #combine
                        node = math.floor(node/2)
                        depth = depth - 1
            
        msg_decode = ucap[n,:]
        
        #Extract the reconciliated codewords
        recon_code_one = msg_cap[-K:]
        recon_code_two = msg_decode[Q1[-K:]-1]

        return  recon_code_one, recon_code_two

    def CRC_Decoder(self,msg_cap, frozen_values, error_prob,polar_code) ->Tuple[bool,np.ndarray,np.ndarray,bool,int]:
        """Implements Successive Cancellation Decoder for Polar Codes
        Args:
            msg_cap (np.ndarray): Message part of the syndrome, must be moved outside for online reconciliation
            frozen_values (np.ndarray) : Frozen values of the syndrome used to dcode the polar code
            error_prob (float): Error Probability
            polar_code (np.ndarray): Polar code to be decoded

        Returns:
            Success_recon (bool) : Flag indicating reconciliation succeeded or not
            recon_code_one (np.ndarray) : First reconciliated code
            recon_code_two (np.ndarray) : Second reconciliated code
            cant_reconcile (bool) : Flag indicating reconciliation cannot be done
            Nerrs (int) : Number of errors after reconciliation
        """
        
        n = self.n
        N = self.N
        Q1 = self.Q1
        K = self.K
        F = self.F
        A = self.A
        crcg = self.crcg
        crcL = self.crcL
        rmax = self.rmax
        cant_reconcile  = 0 
        u = np.zeros(N)
        nL = 8  #16  #8  ################################################

        frozen_indices = np.array(Q1[:N-K])-1
        msg_indices = np.array(Q1[N-K:])-1
        u[frozen_indices] = frozen_values
        u[msg_indices] = msg_cap
        u_source = u
        error_prob_ratio = (1-error_prob) / error_prob
        r = (1 - 2*polar_code) * error_prob_ratio #belief of root; AWGN channel I

        # for CRC check
        msg_cap_values = np.array(msg_cap[:A])
        q,rem = self.gf2_div(np.array(msg_cap[:A]),crcg)
        
        Syndrom_CRC = np.concatenate((rem,np.zeros(crcL-len(rem))))
        Syndrom_CRC = Syndrom_CRC[::-1]
        Syndrom_CRC = operator.xor(Syndrom_CRC.astype(bool),np.array(msg_cap[A:]).astype(bool))
        
        #quantization
        r = self.satx(np.array(r),rmax)
        
        #rq = np.reshape(np.array(r),(1,np.array(r).shape[0] ))   #round(r/rmax*maxqr)
        rq = np.array(r)
        #nL SC decoders
        LLR = np.zeros((nL,n+1,N)) #beliefs in nL decoders
        ucap = np.zeros((nL,n+1,N)) #decisions in nL decoders
        PML = math.inf*np.ones((nL)) #Path metrics
        PML[0] = 0
        ns = np.zeros((2*N-1)) #node state vector

        LLR[:,0,:]  = np.tile(rq,(nL,1))
        
        DML = np.zeros((nL,N))
        PMLL = np.zeros((nL,N))
        node = 0
        depth = 0
        done = 0
        count_test = 0
        count_DM = np.zeros((2*N-1,1))
        count_DM_val = np.zeros((2*N-1,1))
        while(done == 0):
            if (depth == n):
                DM = np.squeeze(LLR[:,n,node]) #decision metrics
                count_DM[count_test] = loop_check
                count_DM_val[count_test] = DM[0]
                count_test = count_test + 1
                DML[:,node] = DM
                PMLL[:,node] = PML
                if((node+1) in F):  #is node frozen
                    ucap[:,n,node] = u_source[node]
                    if(u_source[node] == 0):
                        PML = PML + np.abs(DM)*(DM < 0)  #if DM is negative, add |DM|
                    else:
                        PML = PML + np.abs(DM)*(DM > 0)  #if DM is negative, add |DM|

                else:
                    dec = DM < 0  #decisions as per DM
                    #print(LLR[:,n,node])
                    PM2 = np.hstack([PML,PML+np.abs(DM)])
                    pos = np.argsort(PM2, kind = 'mergesort') #Mergesort retains the position of unchanged indices
                    pos = pos[:nL]
                    PML = PM2[pos] #In PM2(:), first nL are as per DM, next nL are opposite of DM
                    pos1 = pos > nL-1 #surviving with opposite of DM: 1, if pos is above nL
                    pos[pos1] =  pos[pos1] -(nL)  #adjust index
                    dec = dec[pos] #decisions of the survivors
                    dec[pos1] = 1 - dec[pos1] #flip for opposite of DM
                    LLR = LLR[pos,:, :]  #rearrange the decoder states
                    ucap = ucap[pos, : ,:]
                    ucap[:,n,node] = dec


                if (node == N-1):
                    done = 1
                else:
                    node = math.floor(node /2)
                    depth = depth -1
        
            else:
                #nonleaf
                npos = 2**depth -1 + node #position of node in node state vector
                if(ns[npos] == 0):
                    temp = 2 **(n-depth)
                    Ln = np.squeeze(LLR[:,depth,temp*node:temp*(node+1)]) #incoming beliefs
                    a = Ln[:,:int(temp/2)]
                    b = Ln[:,int(temp/2):]
                    node = node*2
                    depth = depth +1
                    temp = int(temp /2)
                    LLR[:,depth,temp*node:temp*(node+1)] = self.f_crc(a,b)
                    minsum = self.f_crc(a,b)
                    ns[npos] = 1
                    loop_check = 1
                else:
                    if(ns[npos] == 1):
                        temp = 2 **(n-depth)
                        Ln = np.squeeze(LLR[:,depth,temp*node:temp*(node+1)]) #incoming beliefs
                        a = Ln[:,:int(temp/2)]
                        b = Ln[:,int(temp/2):]
                        lnode = 2*node
                        ldepth = depth + 1
                        ltemp = int(temp / 2)
                        ucapn = np.squeeze(ucap[:,ldepth,ltemp*lnode:ltemp*(lnode+1)])
                        node = node * 2 + 1
                        depth = depth + 1
                        temp = int(temp / 2)
                        LLR[:,depth,temp*node:temp*(node+1)] = self.g_crc(a,b,ucapn)
                        temp_test = self.g_crc(a,b,ucapn)
                        loop_check = 2
                        ns[npos] = 2

                    else:
                        temp = 2 **(n-depth)
                        lnode = 2*node
                        rnode = 2*node+1
                        cdepth = depth +1
                        ctemp = int(temp/2)
                        ucapl = np.squeeze(ucap[:,cdepth,ctemp*lnode:ctemp*(lnode+1)])
                        ucapr = np.squeeze(ucap[:,cdepth,ctemp*rnode:ctemp*(rnode+1)])
                        if(ucapl.ndim == 1):
                            ucapl = np.reshape(ucapl,(ucap.shape[0],1))
                        if(ucapr.ndim ==1):
                            ucapr = np.reshape(ucapr,(ucapr.shape[0],1))
                        ucap[:,depth,temp*node:temp*(node+1)] = np.hstack(((ucapl+ucapr)%2,ucapr))
                        node = math.floor(node / 2)
                        depth = depth -1

        msg_decode = np.squeeze(ucap[:,n,:])
        
        cout = -1
        Q_list = [x - 1 for x in Q1[N-K:]]
        for c1 in range(nL):
            info_part = msg_decode[c1,Q_list]
            q,r1 = self.gf2_div(info_part[:A],crcg)
            Syndrom_check = np.concatenate((r1,np.zeros(crcL-len(r1))))
            Syndrom_check = Syndrom_check[::-1]
            Syndrom_check = operator.xor(Syndrom_check.astype(bool),np.array(info_part[A:]).astype(bool))
            if (Syndrom_CRC == Syndrom_check).all():
                cout = c1
                break
        if(cout == -1):
            cant_reconcile = 1
            cout = 0

        msg_decode = msg_decode[cout,:]

        recon_code_one = msg_cap[-K:(-K+A-1)]
        recon_code_two = msg_decode[np.array(Q1[-K:(-K+A-1)])-1]

        
        return recon_code_one,recon_code_two,cant_reconcile

    def f_crc(self,a,b) ->np.ndarray:
        """Implements minsum function for CRC decoder
        Args:
            a (np.ndarray) : First array
            b (np.ndarray) : Second array
        Returns 
            f_crc(np.ndarray) : Minsum array
        """

        r_a,c_a = a.shape
        r_b,c_b = b.shape
        f_crc = []
        if(a.shape == b.shape):
            for i in range(r_a):
                f = []
                for j in range(c_a):
                    f.append((1-2*(a[i][j]<0))*(1-2*(b[i][j]<0))*min(abs(a[i][j]),abs(b[i][j])))
                f_crc.append(f)
        else:
            raise ValueError("Shapes of a and b must be equal")
        
        return np.array(f_crc)

    def g_crc(self,a,b,c) -> np.ndarray:
        """ Implements g function for CRC decoder
        Args:
            a (np.ndarray) : First array
            b (np.ndarray) : Second array
            c (np.ndarray) : Third array
        Returns 
            g_crc(np.ndarray) : Minsum array
        """
        r_a,c_a = a.shape
        r_b,c_b = b.shape
        #r_c,c_c = c.shape 
        c = np.reshape(c,(r_a,c_a)) 
        g_crc = []
        if(a.shape == b.shape):  
            for i in range(r_a):
                g = []
                for j in range(c_a):
                    satx = self.satx((b[i][j]+(1-2*c[i][j])*a[i][j]),self.maxqr)
                    g.append(satx)
                g_crc.append(g)
        else:
            raise ValueError("Shapes of a,b and c must be equal")
        
        return np.array(g_crc)

    def g(self,a,b,c) -> np.ndarray:
        """ Implements g function for SC decoder
        Args:
            a (np.ndarray) : First array
            b (np.ndarray) : Second array
            c (np.ndarray) : Third array
        Returns 
            g_value (np.ndarray) : Minsum array
        """
        length_a = len(a)
        length_b = len(b)
        length_c = len(c)
        g_value = []
        if(((length_a == length_b) & (length_b == length_c))):
            for i in range(length_a):
                g_value.append(b[i]+(1-2*c[i])*a[i])
        else:
            raise ValueError("a,b,c must be of equal length")
        
        return g_value

    def f(self,a,b) -> np.ndarray:
        """Implements minsum function for SC decoder
        Args:
            a (np.ndarray) : First array
            b (np.ndarray) : Second array
        Returns 
            f_value (np.ndarray) : Minsum array
        """
    
        length_a = len(a)
        length_b = len(b)
        f_value = []
        
        if(length_a == length_b):
            for i in range(length_a):
                f_value.append((1-2*(a[i]<0))*(1-2*(b[i]<0))*min(abs(a[i]),abs(b[i])))
        else:
            raise ValueError("a,b must be of equal length")
        
        return np.array(f_value)

    def satx(self,x,th) -> np.ndarray:
        """Saturation function
        Args:
            x (np.ndarray) : Input array
            th (float) : Saturation threshold

        Returns:
            satx (np.ndarray) : Saturation array
        """
        x_shape = x.shape
        if(x_shape == ()):
            satx = min(max(x,-th),th)
        else:
            satx = []
            for i in range(x_shape[0]):
                satx.append(min(max(x[i],-th),th))
        
        return satx

    def strip_zeros(self,a) -> np.ndarray:
        """Strip un-necessary leading (rightmost) zeroes
        from a polynomial"""

        return np.trim_zeros(a, trim='b')

    def gf2_div(self,dividend, divisor) -> Tuple[np.ndarray, np.ndarray]:

        """Applies GF2 division between two poynomials
            Args: 
            dividend (np.ndarray) -> Dividend array polynomial
            divisor (np.ndarray) -> Divisor array polynomial

            Returns:
            q (np.ndarray) -> quotient polynomial
            r (np.ndarray) -> remainder polynomial
        """
        N = len(dividend) - 1
        D = len(divisor) - 1

        if dividend[N] == 0 or divisor[D] == 0:
            dividend, divisor = self.strip_zeros(dividend), self.strip_zeros(divisor)

        if not divisor.any():  # if every element is zero
            raise ZeroDivisionError("polynomial division")
        elif D > N:
            q = np.array([])
            return q, dividend

        else:
            u = dividend.astype("uint8")
            v = divisor.astype("uint8")

            m = len(u) - 1
            n = len(v) - 1
            scale = v[n].astype("uint8")
            q = np.zeros((max(m - n + 1, 1),), u.dtype)
            r = u.astype(u.dtype)

            for k in range(0, m - n + 1):
                d = scale and r[m - k].astype("uint8")
                q[-1 - k] = d
                r[m - k - n:m - k + 1] = np.logical_xor(r[m - k - n:m - k + 1], np.logical_and(d, v))
 
            r = self.strip_zeros(r)

        return q, r



    




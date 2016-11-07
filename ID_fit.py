#comments and questions can be send to daniele.granata@gmail.com

import sys,argparse
import numpy as n
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix
from scipy.spatial import distance
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.utils.graph import graph_shortest_path

def func(x,a,b,c):
    return a*n.log(n.sin(x/1*n.pi/2.))
         
def func2(x,a):
    return -a/2.*(x-1)**2

def func3(x,a,b,c):
    return n.exp(c)*n.sin(x/b*n.pi/2.)**a

def main(argv):
     
     parser = argparse.ArgumentParser(epilog="NOTE: it is important to have a smooth histogram for accurate fitting\n\n")
     parser.add_argument("filename", help="input filename")
     
     parser.add_argument("-m", "--metric" , type=str,  help="define the scipy distance to be used   (Default: euclidean or hamming for MSA)",default='euclidean')
     parser.add_argument("-x", "--matrix", help="if the input file contains already the complete upper triangle of a distance matrix (2 Formats: (idx_i idx_j distance) or simply distances list ) (Opt)", action="store_true")
     parser.add_argument("-k", "--n_neighbors", type=int, help="nearest_neighbors parameter (Default k=3)", default=3)
     parser.add_argument("-r", "--radius", type=float, help="use neighbor radius instead of nearest_neighbors  (Opt)")
     parser.add_argument("-b", "--n_bins", type=int, help="number of bins for distance histogram (Default 50)",default=50)
     parser.add_argument("-M", "--r_max", type=float, help="fix the value of distance distribution maximum in the fit (Opt)",default=0)
     parser.add_argument("-n", "--r_min", type=float, help="fix the value of shortest distance considered in the fit (Opt, -1 force the standard fit, avoiding consistency checks)",default=-10)
     parser.add_argument("-D", "--direct", help="analyze the direct (not graph) distances (Opt)", action="store_true")
     parser.add_argument("-I", "--projection", help="produce an Isomap projection using the first ID components (Opt)", action="store_true")
     
     args = parser.parse_args()
     #print args
     input_f = args.filename
     me=args.metric
     n_neighbors = args.n_neighbors
     radius=args.radius
     MSA=False
     n_bins = args.n_bins
     rmax=args.r_max
     mm=-10000

     print '\nFile name: ', input_f
     
     #0 Reading input file
     f1 = open(input_f)
     data = []
     data_line = []
     labels = []

     for line in f1:
         if line[0]==">" : 
               MSA=True
               labels.append(line)
         if line[0]!=">" and MSA==True : 
               data.append([ord(x) for x in line[:-1]])
               data_line.append(line)
         elif line[0]!="#" and MSA==False : 
               data.append([float(x) for x in line.split()])
               data_line.append(line) 
     f1.close()

     data = n.asarray(data)
     if MSA : me='hamming'
     if args.matrix : me='as from the input file'
     print 'Metric: ', me
     if radius>0. and (args.direct==False) : print 'Nearest Neighbors Radius:', radius
     elif (args.direct==False): print 'Nearest Neighbors number K: ', n_neighbors
     else : print 'Distance distribution are calculated based on the  direct input-space distances '
     
     if radius>0. :  
        filename = str(input_f.split('.')[0])+'R'+str(radius)
     else  :
        filename = str(input_f.split('.')[0])+'K'+str(n_neighbors)
     #0
      
     #1 Computing geodesic distance on connected points of the input file and relative histogram
     if args.matrix :
        if data.shape[1] == 1 :
           dist_mat=distance.squareform(data.ravel())
           mm=dist_mat.shape[1]
        elif data.shape[1] == 3 : 
           mm=int(max(data[:,1]))
           dist_mat=n.zeros((mm,mm))
           for i in range(0,data.shape[0]):
               dist_mat[int(data[i,0])-1,int(data[i,1])-1]=data[i,2]
               dist_mat[int(data[i,1])-1,int(data[i,0])-1]=data[i,2]
        else : print 'ERROR: The distances input is not in the right matrix format' ; sys.exit(2)

        print "\n# points: ", mm

        A=n.zeros((mm,mm))
        rrr=[]
           
        if args.direct : C=dist_mat
        if radius > 0. :
           for i in range(0,mm):
               ll=dist_mat[i] < radius
               A[i,ll]=dist_mat[i,ll]
        else :
           rrr=n.argsort(dist_mat)
           for i in range(0,mm):
               ll=rrr[i,0:n_neighbors+1]
               A[i,ll]=dist_mat[i,ll]
           radius = A.max()
        C= graph_shortest_path(A,directed=False)
        
     else : 
        print "\n# points, coordinates: ", data.shape
        if args.direct : C=distance.squareform(distance.pdist(data,me));
        elif radius>0. :
           A = radius_neighbors_graph(data, radius,metric=me,mode='distance')
           C= graph_shortest_path(A,directed=False)
        else  :
           A = kneighbors_graph(data, n_neighbors,metric=me,mode='distance')
           C= graph_shortest_path(A,directed=False)
           radius=A.max()

     C=n.asmatrix(C)
     connect=n.zeros(C.shape[0])
     conn=n.zeros(C.shape[0])
     for i in range(0,C.shape[0]) :
         conn_points=n.count_nonzero(C[i])
         conn[i]=conn_points
         if conn_points > C.shape[0]/2. : connect[i]=1
         else : C[i]=0

     if n.count_nonzero(connect) > C.shape[0]/2. :
        print 'Number of connected points:', n.count_nonzero(connect), '(',100*n.count_nonzero(connect)/C.shape[0],'% )'
     else : print 'The neighbors graph is highly disconnected, increase K or Radius parameters' ; sys.exit(2)

     if n.count_nonzero(connect) < data.shape[0] :
        data_connect_file = open('connected_data_{0}.dat'.format(filename), "w")
        for i in range(0,C.shape[0]) :
            if connect[i]==1 :
               if MSA : data_connect_file.write(labels[i])
               data_connect_file.write(data_line[i])
        data_connect_file.close()

     
     indices = n.nonzero(n.triu(C,1))
     dist_list = n.asarray( C[indices] )[-1]

     h=n.histogram(dist_list,n_bins)
     dx=h[1][1]-h[1][0]


     plt.figure(1)
     plt.plot(h[1][0:n_bins]+dx/2,h[0],'o-',label='histogram')
     plt.xlabel('r')
     plt.ylabel('N. counts')
     plt.legend()
     plt.savefig(filename+'_hist.png')
     distr_x = []
     distr_y = []

     avg=n.mean(dist_list)
     std=n.std(dist_list)

     if rmax> 0 : 
        avg=rmax
        print '\nNOTE: You fixed r_max for the initial fitting, average will have the same value' 
     else : 
        mm=n.argmax(h[0])
        rmax=h[1][mm]+dx/2

     if args.r_min>= 0 : print '\nNOTE: You fixed r_min for the initial fitting: r_min = ',args.r_min
     if args.r_min== -1 : print '\nNOTE: You forced r_min to the standard procedure in the initial fitting'

     print '\nDistances Statistics:'
     print 'Average, standard dev., n_bin, bin_size, r_max, r_NN_max:', avg , std, n_bins, dx, rmax, radius,'\n'
     #1
     tmp=1000000
     if(args.r_min>=0) : tmp=args.r_min
     elif(args.r_min==-1) : tmp=rmax-std
 
     if(n.fabs(rmax-avg)>std) :
        print 'ERROR: There is a problem with the r_max detection:' 
        print '       usually either the histogram is not smooth enough (you may consider changing the n_bins with option -b)'
        print '       or r_max and r_avg are too distant and you may consider to fix the first detection of r_max with option -M' 
        print '       or to change the neighbor parameter with (-r/-k)'
        plt.show()
        sys.exit()

     elif(rmax<= min(radius+dx,tmp)) :
        print 'ERROR: There is a problem with the r_max detection, it is shorter than the largest distance in the neighbors graph.'
        print '       You may consider to fix the first detection of r_max with option -M and/or the r_min with option -n to fix the fit range' 
        print '       or to decrease the neighbors parameter with (-r/-k)'
        plt.show()
        sys.exit()

     #2 Finding actual r_max and std. dev. to define fitting interval [rmin;rM] 
     distr_x=h[1][0:n_bins]+dx/2
     distr_y=h[0][0:n_bins]
     
     res= n.empty(25)
     left_distr_x = n.empty(n_bins)
     left_distr_y = n.empty(n_bins)
     left_distr_x= distr_x[n.logical_and(distr_x[:]>rmax-std, distr_x[:]<rmax+std/2.0)]
     left_distr_y= n.log(distr_y[n.logical_and(distr_x[:]>rmax-std, distr_x[:]<rmax+std/2.0)])
     coeff = n.polyfit(left_distr_x,left_distr_y,2,full='False')    
     a0=coeff[0][0]
     b0=coeff[0][1]
     c0=coeff[0][2]
     
     rmax = -b0/a0/2.0
     if(args.r_max>0) : rmax=args.r_max 
     std=n.sqrt(-1/a0/2.)
     left_distr_x= distr_x[n.logical_and(distr_x[:]>rmax-std, distr_x[:]<rmax+std/2.)]
     left_distr_y= n.log(distr_y[n.logical_and(distr_x[:]>rmax-std, distr_x[:]<rmax+std/2.)])
     coeff = n.polyfit(left_distr_x,left_distr_y,2,full='False')
     a=coeff[0][0]
     b=coeff[0][1]
     c=coeff[0][2]
     
     rmax_old=rmax
     std_old=std
     rmax = -b/a/2.
     std=n.sqrt(-1/a/2.)   # it was a0
     rmin=max(rmax-2*n.sqrt(-1/a/2.)-dx/2,0.)
     if(args.r_min>=0) : 
        rmin=args.r_min
     elif (rmin < radius and args.r_min!=-1) : 
        rmin = radius 
        print '\nWARNING: For internal consistency r_min has been fixed to the largest distance (r_NN_max) in the neighbors graph.'
        print '         It is possible to reset the standard definition of r_min=r_max-2*sigma running with option "-n -1" ' 
        print '         or you can use -n to manually define a desired value (Example: -n 0.1)\n' 
          
     rM=rmax+dx/4
 
     if(n.fabs(rmax-rmax_old)>std_old/4 ) :    #fit consistency check
       print '\nWARNING: The histogram is probably not smooth enough (you may try to change n_bin with -b), rmax is fixed to the value of first iteration\n'  
       #print rmax,rmax_old,std/4,std_old/4
       rmax=rmax_old
       a=a0
       b=b0
       c=c0
       if(args.r_min>=0) :
          rmin=args.r_min
       elif (rmin < radius and args.r_min!=-1) :
          rmin = radius
          print '\nWARNING2: For internal consistency r_min has been fixed to the largest distance in the neighbors graph (r_NN_max).'
          print '          It is possible to reset the standard definition of r_min=r_max-2*sigma running with option "-n -1" '
          print '          or you can use -n to manually define a desired value (Example: -n 0.1)\n'
       rM=rmax+dx/4
     #2

     #3 Gaussian Fitting to determine ratio R
     
     left_distr_x= distr_x[n.logical_and(n.logical_and(distr_x[:]>rmin,distr_x[:]<=rM),distr_y[:]>0.000001)]/rmax
     left_distr_y= n.log(distr_y[n.logical_and(n.logical_and(distr_x[:]>rmin,distr_x[:]<=rM),distr_y[:]>0.000001)])-(4*a*c-b**2)/4./a

     fit =  curve_fit(func2,left_distr_x,left_distr_y)
     ratio=n.sqrt(fit[0][0])
     y1=func2(left_distr_x,fit[0][0])
     #3

     #4 Geodesics D-Hypersphere Distribution Fitting to determine Dfit

     fit = curve_fit(func,left_distr_x,left_distr_y)
     Dfit=(fit[0][0])+1


     y2=func(left_distr_x,fit[0][0],fit[0][1],fit[0][2])
     #4

     
     #5 Determination of Dmin

     D_file = open('D_residual_{0}.dat'.format(filename), "w")
     
     for D in range(1,26):
         y=(func(left_distr_x,D-1,1,0))
         for i in range(0, len(y)):
             res[D-1] = n.linalg.norm((y)-(left_distr_y))/n.sqrt(len(y))
         D_file.write("%s " % D)
         D_file.write("%s\n" % res[D-1])

     Dmin = n.argmax(-res)+1

     y=func(left_distr_x,Dmin-1,fit[0][1],0)
     #5

     #6 Printing results
     print '\nFITTING PARAMETERS:' 
     print 'rmax, std. dev., rmin', rmax,std,rmin
     print '\nFITTING RESULTS:' 
     print 'R, Dfit, Dmin', ratio,Dfit,Dmin , '\n'

     if(Dmin == 1) : print 'NOTE: Dmin = 1 could indicate that the choice of the input parameters is not optimal or simply an underestimation of a 2D manifold\n'
     fit_file= open('fit_{0}.dat'.format(filename), "w")

     for i in range(0, len(y)):
         fit_file.write("%s " % left_distr_x[i])
         fit_file.write("%s " % ((left_distr_y[i])))
         fit_file.write("%s " % ((y1[i])))
         fit_file.write("%s " % ((y2[i])))
         fit_file.write("%s\n" % ((y[i])))
     fit_file.close() 

             
     stat_file= open('statistics_{0}.dat'.format(filename), "w")
     statistics = str('# Npoints, rmax, standard deviation, R, D_fit, Dmin \n# \
     {}, {}, {}, {}, {}, {}\n'.format(n.count_nonzero(connect),rmax,std,ratio,Dfit,Dmin))
     stat_file.write("%s" % statistics)
     for i in range(0, len(distr_x)-2): 
	 stat_file.write("%s " % distr_x[i])
	 stat_file.write("%s " % distr_y[i])
	 stat_file.write("%s\n" % n.log(distr_y[i]))
     stat_file.close()
     
     plt.figure(2)
     plt.plot(left_distr_x,left_distr_y,'o-',label=str(input_f.split('.')[0]))
     plt.plot(left_distr_x,y1,label='Gaussian fit for R ratio')
     plt.plot(left_distr_x,y2,label='D-Hypersphere Fit for D_fit')
     plt.plot(left_distr_x,y,label='D_min-Hypersphere Distribution')
     plt.xlabel('r/r$_{max}$')
     plt.ylabel('log p(r)/p(r$_{max}$)')
     plt.legend(loc=4)
     plt.savefig(str(input_f.split('.')[0])+'_fit.png')  
     

     plt.figure(3)
     plt.plot(range(1,26),res,'o-',label=str(input_f.split('.')[0])+' D_min')
     plt.legend()
     plt.xlabel('D')
     plt.ylabel('RMDS')
     plt.show()
     plt.savefig(str(input_f.split('.')[0])+'_Dmin.png')


     #6
   
     #7 Optional: Isomap projection
     if args.projection :
        from sklearn.decomposition import KernelPCA
        C2=(distance.squareform(dist_list))**2
        C2=-.5*C2
        obj_pj=KernelPCA(n_components=100,kernel="precomputed")
        proj=obj_pj.fit_transform(C2)
        n.savetxt('proj_'+str(input_f.split('.')[0])+'.dat',proj[:,0:Dmin])
     print 'NOTE: it is important to have a smooth histogram for accurate fitting\n'

if __name__ == "__main__":
     main(sys.argv[1:])

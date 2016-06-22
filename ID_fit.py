import sys, getopt
import numpy as n
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
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

     if (len(sys.argv) == 1): 
         print 'Usage: fit_find_D_file.py -f <filename> (Optional: -m <metric>)'
         sys.exit(2)

     input_f = ''
     me='euclidean'
     n_neighbors = 3
     radius=0
     MSA=False
     direct=False
     isomap_projection=False
     n_bins = 0
     rmax=0

     try:
         opts, args = getopt.getopt(argv,"hf:m:k:r:M:D:b:I",["input_f=","metric=","nearest_neighbors","neighbor_radius","r_max","direct","nbin","project"])
     except getopt.GetoptError:
         print 'Usage: fit_find_file.py -f <filename> OR -h'
         sys.exit(2)
     for opt, arg in opts:
         if opt == '-h':
             print 'Usage: fit_find_file.py -f <filename> '
             print '                        -m <metric>   (Default: euclidean or hamming for MSA)'
             print '                        -k <nearest_neighbors parameter>   (Default k=3)'
             print '                        -r <neighbor_radius>   (Optional, instead of -k)'
             print '                        -D <analysis of direct (not graph) distances>   (Optional)'
             print '                        -b <number of bins for distance histogram>   (Default 50)'
             print '                        -I <Isomap projection>   (Optional)'
             print '\nNOTE: it is important to have a smooth histogram for accurate fitting'
             sys.exit()
         elif opt in ("-f", "--input_f"):
             input_f = arg
         elif opt in ("-m", "--metric"):
             me = arg
         elif opt in ("-k", "--nearest_neighbors"):
             n_neighbors = int(arg)
         elif opt in ("-r", "--neighbor_radius"):
             radius = float(arg)
         elif opt in ("-M", "--r_max"):
             rmax = float(arg)
         elif opt in ("-D", "--direct"):
             direct=True
         elif opt in ("-b", "--nbin"):
             n_bins = int(arg)
         elif opt in ("-I", "--project"):
             isomap_projection = True 

     n_bins=int(n_bins)

     print '\nFile name is ', input_f
     
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
               data.append(line.split())
               data_line.append(line) 
     f1.close()

     data = n.asarray(data)
     if MSA : me='hamming'
     print 'Metric is ', me

     if radius>0. and (direct==False) : print 'Radius is', radius
     elif (direct==False): print 'K is ', n_neighbors
     else : print 'Distance distribution are calculated based on the  direct input-space distances '
     print "# points, coordinates: ", data.shape
     
     if radius>0. :  
        filename = str(input_f.split('.')[0])+'R'+str(radius)
     else  :
        filename = str(input_f.split('.')[0])+'K'+str(n_neighbors)
     #0
 
     #1 Computing geodesic distance on connected points of the input file and relative histogram
     if direct : C=distance.squareform(distance.pdist(data,me)); 
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
     

     if n.count_nonzero(connect) > data.shape[0]/2. :
        print 'Number of connected points:', n.count_nonzero(connect), '(',100*n.count_nonzero(connect)/data.shape[0],'% )'
     else : print 'The neighbors graph is highly disconnected, increase K or Radius parameters' ; sys.exit(2)

     if n.count_nonzero(connect) < data.shape[0] :
        data_connect_file = open('connected_data_{0}.dat'.format(filename), "w")
        for i in range(0,data.shape[0]) :
            if connect[i]==1 :
               if MSA : data_connect_file.write(labels[i])
               data_connect_file.write(data_line[i])
        data_connect_file.close()
 
     indices = n.nonzero(n.triu(C,1))
     dist_list = n.asarray( C[indices] )[-1]

     if n_bins==0 : n_bins=50 
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
        print '\n You fixed rmax, average will have the same value' 
     else : 
        mm=n.argmax(h[0])
        rmax=h[1][mm]+dx/2

     print '\nDistances Statistics:'
     print 'Average, standard dev., n_bin, bin_size, rmax:', avg , std, n_bins, dx, rmax,'\n'
     #1

     if(n.fabs(rmax-avg)>std or rmax< radius+dx ) : 
        
        print 'ERROR: There is a problem with the r_max detection: \nusually either probably the histogram is not smooth enough (you may consider changing the n_bins with option -b)\nor r_max and r_avg are too distant (you may consider to fix the first detection of r_max with option -M or the neighbors parameter with (-r/-k)'
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
     std=n.sqrt(-1/a0/2.)
     rmin=max(rmax-2*n.sqrt(-1/a/2.)-dx/2,radius)
     rM=rmax+dx/4

     if(n.fabs(rmax-rmax_old)>std/4 ) :    #fit consistency check
       print '\nWARNING: The histogram is probably not smooth enough (you may try to change n_bin with -b), rmax is fixed to the value of first iteration\n'  
       rmax=rmax_old
       a=a0
       b=b0
       c=c0
       rmin=max(rmax-2*n.sqrt(-1/a/2.)-dx/2,radius)
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
     #plt.show()
     plt.savefig(str(input_f.split('.')[0])+'_Dmin.png')


     #6
   
     #7 Optional: Isomap projection
     if(isomap_projection==True) :
        from sklearn.decomposition import KernelPCA
        C2=(distance.squareform(dist_list))**2
        C2=-.5*C2
        obj_pj=KernelPCA(n_components=100,kernel="precomputed")
        proj=obj_pj.fit_transform(C2)
        #proj_file = open('proj_{0}.dat'.format(filename), "w")
        n.savetxt('proj_'+str(input_f.split('.')[0])+'.dat',proj[:,0:Dmin])
        #proj_file.close()


if __name__ == "__main__":
     main(sys.argv[1:])

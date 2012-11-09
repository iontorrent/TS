# Modify these variables for desired effect
# filename is where the matrix will be written
# I use the convention tik10F58.bin where 10 is 10 times alpha and 58 is the number of frames
filename="/home/ablanchard/Code/tik10F58.bin";
# N is number of frames
N=58;
# alpha is balance between error and smoothing
alpha = 1.0;
#width is number of upper and lower diagonals in final matrix
width=7;
#denominator is for scaling the integer representation of the matrix
denominator = 8192

# DON'T TOUCH ANYTHING BELOW
A=eye(N);
S=tril(ones(N));
D=eye(N) - diag(ones(N-1,1),-1);
D(1,1)=0;
L = D*D;
La=L*alpha;
# make basic matrix
T=inv((A'*A) + (La'*La)) *A';
# here's where we turn it into scaled integers
T_i = round(T*denominator);
# and limit the band-width to diagonal +- width
M=ones(N)-(tril(ones(N),-(width+1)) + triu(ones(N),width+1));
T =M .* T_i;

# here we create the ancillary arrays of summation limits
for i=1:N
  if(i<=width+1)
    i_low(i) = 0;
  else
    i_low(i) = i-width-1;
  endif
  if(i>=N-width)
    i_high(i) = N-1;
  else
    i_high(i) = i+width-1;
  endif
endfor

# now we write it out in the format used by Analysis
tik_fid = fopen(filename, "w");
fwrite(tik_fid, 0XABCDBEEF, "uint32",0,"ieee-le");
fwrite(tik_fid, N, "int32",0,"ieee-le");
fwrite(tik_fid, denominator, "int32",0,"ieee-le");
# this should be it for the header, now for the summation limits
fwrite(tik_fid, i_low(1:N), "int32",0,"ieee-le");
fwrite(tik_fid, i_high(1:N), "int32",0,"ieee-le");
# and now the matrix itself
fwrite(tik_fid, T(1:N,1:N), "int32",0,"ieee-le");
fclose(tik_fid);


img = readim('https://i.stack.imgur.com/Zq1f4.jpg');
#Simplify using non-linear diffusion
s = colordiffusion(img,2);
# Find lines -- the positive response of the Laplace operator
l = laplace(s,1.5);
l = tensorfun('immax',l);
l = stretch(clip(l,0.4,4),0,100,1,0);
#Reemove short lines
l = pathopening(l,8,'closing','constrained');
#Simplify color image using diffusion and k-means clustering
s = colordiffusion(gaussf(img),5);
s = quantize(s,10,'minvariance');
s = gaussf(s);
# Paint lines on simplified image
out = s * l;

# Color diffusion:
function out = colordiffusion(out,iterations)sigma = 0.8;
K = 10;
for ii = 1:iterations
grey = colorspace(out,'grey');
nabla_out = gradientvector(grey,sigma);
D = exp(-(norm(nabla_out)/K)^2);
out = out + divergence(D * nabla_out);
end
end
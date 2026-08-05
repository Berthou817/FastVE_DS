function [Solution1, QQ1,Solution2, QQ2] = Biot_fast(etaf_k,Kd,Gu,Tor,phi,alpha,B,M,rhof,rhot,G,f)

omega = f*2*pi;
Pe = 1/etaf_k;
c11         = Kd+ 4/3*Gu;
Tor_fi = Tor/phi;
iM_ELan1    = [  (alpha/B + 4/3*Gu/M), alpha; alpha,  1]./ (  alpha/B + 4/3*G/M - alpha^2) .* c11;
rho_ft  = rhof/rhot; rho_at = rhof*Tor_fi/rhot;
Mdvp1       = rhot.*[1, -rho_ft; -rho_ft, +rho_at];
iMdvp1      = inv(Mdvp1);
A11 = iM_ELan1(1,1);    A12 = iM_ELan1(1,2);   A22 = iM_ELan1(2,2);
R11 = iMdvp1(1,1); R12 = iMdvp1(1,2);R22 = iMdvp1(2,2);
beta_d          = 1/Kd;
G0 = Gu;K_u            = 1/beta_d./(1 - B.*alpha );
I1 = rhot; I2 = 1/Pe;  S = 1/c11; a = alpha; aA = alpha/B*(1 + 4/3*G0/K_u);
R12_20 = rho_ft; R22_20 = rho_at;
A0_m = S ^ 2 * I1 ^ 2 * (R12_20 ^ 2 - R22_20) * (a ^ 2 - aA) * omega ^ 4 - 1i * S ^ 2 * I2 * I1 * (a ^ 2 - aA) * omega ^ 3;
A2_m = ( (-2*a * R12_20 + aA * R22_20  + 1) * S * I1 * omega ^ 2) + 1i * I2 * S * aA * omega;
A3_m = 1;

s_squared            =  ( A2_m - (A2_m.*A2_m - 4.*A3_m.*A0_m)^0.5 )./2./A3_m;
QQ1      = imag(s_squared)./real(s_squared);

s_squared2           =  (( A2_m + (A2_m.*A2_m - 4.*A3_m.*A0_m)^0.5 )./2./A3_m );
QQ2      = imag(s_squared2)./real(s_squared2);%.^2
Solution1 = omega*1./real( (( A2_m - (A2_m.*A2_m - 4.*A3_m.*A0_m)^0.5 )./2./A3_m ).^0.5);
Solution2 = omega*1./real( (( A2_m + (A2_m.*A2_m - 4.*A3_m.*A0_m)^0.5 )./2./A3_m ).^0.5);
end
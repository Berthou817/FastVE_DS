function [phi,Rog,Vx_BG,Pt,Pf]=init_data3D_v25( vrho_x_11, name, mult, nx_l,ny_l,nz_l,dims_x,dims_y,dims_z,machineformat,PRECIS)

% This is a function to cut the original material matrix into cut subcubes
% for MPI-GPU Implementation

% The original script is written by Yury Alkhimenkov
% and improved by Zhiyu Hou (UNIL)


% -------------------- knobs --------------------
write    = 1;
plotfig  = 0;            
useParfor= false;        
inclusion= 0;            
ndif     = 0.03;         
% ------------------------------------------------

phi0   = 1e-2;
rhofg  = 1;
k_muf0 = 1;
etas0  = 1;
phiA   = 2*phi0;
rhosg  = 2*rhofg;

lam0   = 2;
Lx     = 30; Ly=Lx; Lz=2*Lx;
arx    = 0.25; ary=0.25; arz=1.0;
lam    = lam0*sqrt(etas0*k_muf0);
RogBG  = rhofg.*phi0 + (1-phi0).*rhosg;

% ---------------- datatype ----------------------
if PRECIS==8
    DAT      = 'double';
    castFunc = @double;
elseif PRECIS==4
    DAT      = 'single';
    castFunc = @single;
else
    error('PRECIS must be 4(single) or 8(double).');
end


% ---------------- MPI sizes ---------------------
mpi_par = true;
NDIMS   = 3; 
nxyz_l  = [nx_l, ny_l, nz_l];
dims    = [dims_x, dims_y, dims_z];
nprocs  = prod(dims);

if mpi_par, COMPUTATION_OVERLAP = 2; else, COMPUTATION_OVERLAP = 0; end
if mpi_par
    nxyz_l = nxyz_l - COMPUTATION_OVERLAP;
    nx_l   = nxyz_l(1); ny_l=nxyz_l(2); nz_l=nxyz_l(3);
    nxyz   = nxyz_l.*dims + COMPUTATION_OVERLAP;
    nx     = nxyz(1); ny=nxyz(2); nz=nxyz(3);
else
    nx=nx_l; ny=ny_l; nz=nz_l;
end

nxyz_l_phi   = [(nx_l  ) (ny_l  ) (nz_l  )] + COMPUTATION_OVERLAP;
nxyz_l_Vx_BG = [(nx_l+1) (ny_l  ) (nz_l  )] + COMPUTATION_OVERLAP;
nxyz_l_Pt    = [(nx_l  ) (ny_l  ) (nz_l  )] + COMPUTATION_OVERLAP;
nxyz_l_Pf    = nxyz_l_Pt;

vrho_x_11_h = castFunc( vrho_x_11 + vrho_x_11.*mult );

phi   = castFunc( zeros( nx, ny, nz ) );
Vx_BG = castFunc( zeros( nx+1, ny, nz ) );   
Pt    = castFunc( zeros( nx, ny, nz ) );
Pf    = castFunc( zeros( nx, ny, nz ) );

coords = gen_coords(nprocs, dims);

% ---------------- grid --------------------------
dx = Lx/nx; dy = Ly/ny; dz = Lz/nz;
xc = (dx/2) : dx : (Lx-dx/2);
yc = (dy/2) : dy : (Ly-dy/2);
zc = (dz/2) : dz : (Lz-dz/2);
[Xc,Yc,Zc] = ndgrid(xc,yc,zc);

if inclusion==1
    radc = ((Xc - Lx*.5)/lam*arx).^2 + ...
           ((Yc - Ly*.5)/lam*ary).^2 + ...
           ((Zc - Lz*.2)/lam*arz).^2;

    phi = castFunc( phi0*ones(nx,ny,nz) );
    msk = radc<1;
    phi(msk) = phi(msk) + phiA;

    dtD = (min([dx,dy,dz]).^2)/6.1;
    ntD = fix(ndif/dtD);
    for idif=1:ntD
        phi(2:end-1,2:end-1,2:end-1) = phi(2:end-1,2:end-1,2:end-1) + dtD* ( ...
            (phi(3:end,  2:end-1,2:end-1) - 2*phi(2:end-1,2:end-1,2:end-1) + phi(1:end-2,2:end-1,2:end-1))/dx^2 + ...
            (phi(2:end-1,3:end,  2:end-1) - 2*phi(2:end-1,2:end-1,2:end-1) + phi(2:end-1,1:end-2,2:end-1))/dy^2 + ...
            (phi(2:end-1,2:end-1,3:end  ) - 2*phi(2:end-1,2:end-1,2:end-1) + phi(2:end-1,2:end-1,1:end-2))/dz^2 );
    end
else
    anom = castFunc( rand(nx,ny,nz) );

    A = reshape(anom, nx*ny, nz);              
    A = cumsum(A,2);
    A = detrend(A.').';                        
    anom = reshape(A, nx, ny, nz);

    A = permute(anom, [1 3 2]);                 
    A = reshape(A, nx*nz, ny);                 
    A = cumsum(A,2);
    A = detrend(A.').';                         
    anom = ipermute( reshape(A, nx, nz, ny), [1 3 2] );

    A = permute(anom, [2 3 1]);                 
    A = reshape(A, ny*nz, nx);                  
    A = cumsum(A,2);
    A = detrend(A.').';
    anom = ipermute( reshape(A, ny, nz, nx), [2 3 1] );

    radius = exp( - ((Xc - Lx*.5)/(Lx/3)).^2 ...
                  - ((Yc - Ly*.5)/(Ly/3)).^2 ...
                  - ((Zc - Lz*.25)/(Lz/2)).^2 );

    phi = anom .* radius;
    phi = phi - min(phi(:));
    mx  = max(phi(:));
    if mx>0
        phi = castFunc( phi0 + phiA * (phi/mx) );
    else
        phi = castFunc( phi0*ones(nx,ny,nz) );
    end
end


phi0BC = mean(phi(:,end), 'all'); 
yBC    = castFunc( (-Ly/2+dy/2):dy:(Ly/2-dy/2) );
Vx_BG  = castFunc( ones(nx+1,1,nz) ) .* reshape(yBC,1,ny,1);

Rog    = rhofg.*phi + (1-phi).*rhosg - RogBG;
Pt     = repmat( castFunc( cumsum( mean(mean(Rog,1),2 ), 'reverse') * dz ), [nx, ny, 1] );
Pf     = Pt;

write_var_fast(vrho_x_11_h, name  , nxyz_l_phi  , nxyz_l, nprocs, coords, DAT, machineformat, write, useParfor);

function coords = gen_coords(nprocs, dims)
    coords = zeros(nprocs,3);
    p=1;
    for idim1=1:dims(1)         
        for idim2=1:dims(2)
            for idim3=1:dims(3)
                coords(p,:)=[idim1-1 ,idim2-1 ,idim3-1];
                p=p+1;
            end
        end
    end
end

function write_var_fast(A, A_name, nxyz_l_A, nxyz_l0, nprocs, coords, DAT, machineformat, writeFlag, usePar)
    if ~writeFlag, return; end
    xStarts = coords(:,1)*nxyz_l0(1) + 1;
    yStarts = coords(:,2)*nxyz_l0(2) + 1;
    zStarts = coords(:,3)*nxyz_l0(3) + 1;
    xEnds   = xStarts + nxyz_l_A(1) - 1;
    yEnds   = yStarts + nxyz_l_A(2) - 1;
    zEnds   = zStarts + nxyz_l_A(3) - 1;


    buf = zeros(nxyz_l_A(1), nxyz_l_A(2), nxyz_l_A(3), class(A));

    writer = @(p) do_write(p, A, A_name, DAT, machineformat, ...
                           xStarts, xEnds, yStarts, yEnds, zStarts, zEnds, buf);
    if usePar && (exist('parfor','file')==2)
        parfor p=1:nprocs
            writer(p);
        end
    else
        for p=1:nprocs
            writer(p);
        end
    end
end

function do_write(p, A, A_name, DAT, machineformat, ...
                  xStarts, xEnds, yStarts, yEnds, zStarts, zEnds, buf)
    xs=xStarts(p); xe=xEnds(p);
    ys=yStarts(p); ye=yEnds(p);
    zs=zStarts(p); ze=zEnds(p);

    buf(:,:,:) = A(xs:xe, ys:ye, zs:ze);

    isave = '0';
    me    = sprintf('%d',p-1);
    fname = [isave '_' me '_' A_name '.data'];
    fid   = fopen(fname,'w');
    fwrite(fid, buf(:)', DAT, 0, machineformat);
    fclose(fid);
    fprintf('process %d of %d written %s\n', p, numel(xStarts), fname);
end

function write_val_fast(val, val_name, DAT, machineformat, writeFlag)
    if ~writeFlag, return; end
    isave = '0'; me='0';
    fid   = fopen([isave '_' me '_' val_name '.data'],'w');
    fwrite(fid,val, DAT, 0, machineformat);
    fclose(fid);
end

end

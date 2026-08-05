function [Vp, attenuation, info] = compute_Vp_Q_from_Vx( ...
        fname1, fname2, nt, irx1, irx2, dt, dx, freq, do_plot)
% -------------------------------------------------------
% Compute P-wave velocity from two Vx receiver signals
% using sub-sample arrival picking (spline + derivative)
%
% INPUT:
%   fname1, fname2 : receiver files (.res)
%   nt             : number of time samples
%   irx1, irx2     : receiver indices (grid index)
%   dt, dx         : time step and spatial step
%   do_plot        : true / false
%
% OUTPUT:
%   Vp             : P-wave velocity (m/s)
%   info           : struct with picking details
% -------------------------------------------------------

    % -------- read data --------
    Vx1 = read_res(fname1, nt);
    Vx2 = read_res(fname2, nt);
    
    % -------- pick arrival times --------
    [t1, amp1] = pick_arrival(Vx1);
    [t2, amp2] = pick_arrival(Vx2);

    % -------- velocity --------
    Vp = abs((irx2 - irx1) * dx / ((t2 - t1) * dt));
     % -------- attenuation --------
    attenuation          = log(amp1./amp2 )./ (pi.*freq* (t2 - t1) *dt );
    % -------- info --------
    info.t1 = t1;   info.t2 = t2;
    info.amp1 = amp1; info.amp2 = amp2;
% -------- optional plot --------
    if do_plot
        figure;
        subplot(121); plot(Vx1,'k'); hold on;
        xline(t1,'r--','LineWidth',1.5);
        title('Receiver 1'); box on;

        subplot(122); plot(Vx2,'k'); hold on;
        xline(t2,'r--','LineWidth',1.5);
        title('Receiver 2'); box on;
    end

end

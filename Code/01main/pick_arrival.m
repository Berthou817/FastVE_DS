function [t_pick, amp_pick] = pick_arrival(Vx)

    % coarse max
    [~, idx0] = max(Vx);

    % local window
    xx = idx0-3:idx0+3;
    yy = Vx(xx);

    % derivative
    dydx = diff(yy)./diff(xx);
    xc = (xx(1:end-1) + xx(2:end))/2;

    % zero crossing
    ix = find(dydx(1:end-1).*dydx(2:end) < 0, 1);

    % spline refinement
    t_pick = interp1(dydx(ix:ix+1), xc(ix:ix+1), 0, 'spline');

    % refined amplitude
    VVV = interp1(xx, yy, (t_pick-3:t_pick+3), 'spline');
    amp_pick = VVV(4);
end

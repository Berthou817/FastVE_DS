function [t_pick, amp_pick] = pick_two_wavepackets(Vx, split_idx)

    N = length(Vx);

    if split_idx <= 1 || split_idx >= N
        error('split_idx must be between 2 and N-1');
    end

    % 左边找最大峰
    [amp1, idx1_local] = max(Vx(1:split_idx));
    idx1 = idx1_local;

    % 右边找最大峰
    [amp2, idx2_local] = max(Vx(split_idx+1:end));
    idx2 = split_idx + idx2_local;

    idxs = [idx1, idx2];
    t_pick = zeros(1,2);
    amp_pick = zeros(1,2);

    for k = 1:2
        idx0 = idxs(k);

        i1 = max(idx0-3,1);
        i2 = min(idx0+3,N);

        xx = i1:i2;
        yy = Vx(xx);

        % 二次插值细化峰值位置
        p = polyfit(xx, yy, 2);
        t_refine = -p(2)/(2*p(1));
        a_refine = polyval(p, t_refine);

        t_pick(k) = t_refine;
        amp_pick(k) = a_refine;
    end

    % 画图检查
    figure; hold on;
    plot(Vx,'k');
    plot(t_pick, interp1(1:N, Vx, t_pick, 'spline'), ...
        'ro', 'MarkerSize', 10, 'LineWidth', 2);
    xline(split_idx, '--b', 'Split');
    xlabel('Sample index');
    ylabel('Amplitude');
    legend('Vx','Picked peaks','Split');
end
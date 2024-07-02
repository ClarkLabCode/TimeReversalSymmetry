function y = NanSem(x,dim)
    y = sqrt(nanvar(x,[],dim));
    n = sum(~isnan(x),dim);
    y = bsxfun(@rdivide,y,sqrt(n));
end
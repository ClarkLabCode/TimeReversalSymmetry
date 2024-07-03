function [ dx1,dx2,dt1,dt2,glidOrder ] = glidDisp( whichGlid,varDt )

switch whichGlid
    case 1 % div 3
        dx1 = 1;
        dx2 = 1;
        dt1 = 1;
        dt2 = 0;
        glidOrder = 3;
    case 2 % con 3
        dx1 = 1;
        dx2 = 0;
        dt1 = 1;
        dt2 = 1;
        glidOrder = 3;
    case 3 % two-point
        dx1 = 1;
        dt1 = 1;
        dx2 = 0;
        dt2 = 0;
        glidOrder = 2;
    case 4 % elbow
        dx1 = 1;
        dx2 = 0;
        dt1 = 1;
        dt2 = 2;
        glidOrder = 3;
    case 5 % late knight
        dx1 = 1;
        dx2 = 1;
        dt1 = 1;
        dt2 = 2;
        glidOrder = 3;
    case 6 % early knight
        dx1 = 0;
        dx2 = 1;
        dt1 = 1;
        dt2 = 2;
        glidOrder = 3;
    case 7 % extra div
        dx1 = 1;
        dx2 = 1;
        dt1 = 2;
        dt2 = 0;
        glidOrder = 3;
    case 8 % extra con
        dx1 = 1;
        dx2 = 0;
        dt1 = 2;
        dt2 = 2;
        glidOrder = 3;
    case 9 % elbow late break
        dx1 = 1;
        dx2 = 0;
        dt1 = 2;
        dt2 = 3;
        glidOrder = 3;
    case 10 % elbow early break
        dx1 = 1;
        dx2 = 0;
        dt1 = 1;
        dt2 = 3;
        glidOrder = 3;
    case 11 % div variable dt
        dx1 = 1;
        dx2 = 1;
        dt1 = varDt;
        dt2 = 0;
        glidOrder = 3;
    case 12 % con variable dt
        dx1 = 1;
        dx2 = 0;
        dt1 = varDt;
        dt2 = varDt;
        glidOrder = 3;
    case 13 % two-point variable dt
        dx1 = 1;
        dt1 = varDt;
        dx2 = 0;
        dt2 = 0;
        glidOrder = 2;
end

end

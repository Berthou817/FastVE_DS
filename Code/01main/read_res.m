function Vx = read_res(fname, nt)
    id = fopen(fname);
    assert(id>0, ['Cannot open ', fname])
    Vx = fread(id,'double');
    fclose(id);
    Vx = reshape(Vx, nt, 1);
end

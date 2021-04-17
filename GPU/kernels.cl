kernel void accelerate_flow(global float* cells_c1, global float* cells_c3, global float* cells_c5, global float* cells_c6, global float* cells_c7, global float* cells_c8, 
                            global float* obstacles, int second_row_from_top, int nx, float density, float accel)
{
	float w1 = density * accel / 9.0;
	float w2 = density * accel / 36.0;

	int jj = second_row_from_top;

	int ii = get_global_id(0);

	if (!obstacles[ii + jj*nx]
		&& (cells_c3[ii + jj*nx] - w1) > 0.f
		&& (cells_c6[ii + jj*nx] - w2) > 0.f
		&& (cells_c7[ii + jj*nx] - w2) > 0.f)
	{
		cells_c1[ii + jj*nx] += w1;
		cells_c5[ii + jj*nx] += w2;
		cells_c8[ii + jj*nx] += w2;
		cells_c3[ii + jj*nx] -= w1;
		cells_c6[ii + jj*nx] -= w2;
		cells_c7[ii + jj*nx] -= w2;
	}
}


kernel void collision(  global float* cells_c0, global float* cells_c1, global float* cells_c2, global float* cells_c3, global float* cells_c4, global float* cells_c5,
                        global float* cells_c6, global float* cells_c7, global float* cells_c8, global float* tmp_cells_c0, global float* tmp_cells_c1, global float* tmp_cells_c2,
                        global float* tmp_cells_c3, global float* tmp_cells_c4, global float* tmp_cells_c5, global float* tmp_cells_c6, global float* tmp_cells_c7, global float* tmp_cells_c8,
                        global float* obstacles, global float* av_vels, int ny, int nx, float omega, local float* partial_av_vels, int tt)
{
	const size_t i = get_global_id(0);
	const size_t j = get_global_id(1);

	int y_n = (j + 1) % ny;
	int x_e = (i + 1) % nx;
	int y_s = (j == 0) ? (j + ny - 1) : (j - 1);
	int x_w = (i == 0) ? (i + nx - 1) : (i - 1);

	float tmp_0 = cells_c0[i   +   j*nx];
	float tmp_1 = cells_c1[x_w +   j*nx];
	float tmp_2 = cells_c2[i   + y_s*nx];
	float tmp_3 = cells_c3[x_e +   j*nx];
	float tmp_4 = cells_c4[i   + y_n*nx];
	float tmp_5 = cells_c5[x_w + y_s*nx];
	float tmp_6 = cells_c6[x_e + y_s*nx];
	float tmp_7 = cells_c7[x_e + y_n*nx];
	float tmp_8 = cells_c8[x_w + y_n*nx];

    float sqrt_u_sq = 0.f;
    
    if(!obstacles[i + j*nx])
    {
        const float w0 = 4.f / 9.f;  /* weighting factor */
	    const float w1 = 1.f / 9.f;  /* weighting factor */
	    const float w2 = 1.f / 36.f; /* weighting factor */
	    const float omega_c = 1.f - omega;

	    float tmp_158 = tmp_1 + tmp_5 + tmp_8;
	    float tmp_367 = tmp_3 + tmp_6 + tmp_7;
	    float tmp_256 = tmp_2 + tmp_5 + tmp_6;
	    float tmp_478 = tmp_4 + tmp_7 + tmp_8;

	    float local_density = tmp_0 + tmp_256 + tmp_478 + tmp_3 + tmp_1;

	    float u_x = (tmp_158 - tmp_367) / local_density;
	    float u_y = (tmp_256 - tmp_478) / local_density;

	    float u_sq = u_y * u_y + u_x * u_x;

        sqrt_u_sq = sqrt(u_sq);

	    float u0 = 0.f;
	    float u1 =   u_x;        /* east */
	    float u2 =         u_y;  /* north */
	    float u3 = - u_x;        /* west */
	    float u4 =       - u_y;  /* south */
	    float u5 =   u_x + u_y;  /* north-east */
	    float u6 = - u_x + u_y;  /* north-west */
	    float u7 = - u_x - u_y;  /* south-west */
	    float u8 =   u_x - u_y;  /* south-east */

	    float c1 = local_density * w1;
	    float c2 = local_density * w2;
	    float c3 = 1.f - (1.5f * u_sq);

	    float de0 = w0 * local_density * c3;
	    float de1 = c1 * (c3 + u1 * (3.f + u1 * 4.5f));
	    float de2 = c1 * (c3 + u2 * (3.f + u2 * 4.5f));
	    float de3 = c1 * (c3 + u3 * (3.f + u3 * 4.5f));
	    float de4 = c1 * (c3 + u4 * (3.f + u4 * 4.5f));
	    float de5 = c2 * (c3 + u5 * (3.f + u5 * 4.5f));
	    float de6 = c2 * (c3 + u6 * (3.f + u6 * 4.5f));
	    float de7 = c2 * (c3 + u7 * (3.f + u7 * 4.5f));
	    float de8 = c2 * (c3 + u8 * (3.f + u8 * 4.5f));

	    tmp_cells_c0[i + j*nx] = tmp_0 * omega_c + omega * de0;
	    tmp_cells_c1[i + j*nx] = tmp_1 * omega_c + omega * de1;
	    tmp_cells_c2[i + j*nx] = tmp_2 * omega_c + omega * de2;
	    tmp_cells_c3[i + j*nx] = tmp_3 * omega_c + omega * de3;
	    tmp_cells_c4[i + j*nx] = tmp_4 * omega_c + omega * de4;
	    tmp_cells_c5[i + j*nx] = tmp_5 * omega_c + omega * de5;
	    tmp_cells_c6[i + j*nx] = tmp_6 * omega_c + omega * de6;
	    tmp_cells_c7[i + j*nx] = tmp_7 * omega_c + omega * de7;
	    tmp_cells_c8[i + j*nx] = tmp_8 * omega_c + omega * de8;

	}
	else
	{ 
	    tmp_cells_c0[i + j*nx] = tmp_0;
	    tmp_cells_c1[i + j*nx] = tmp_3;
	    tmp_cells_c2[i + j*nx] = tmp_4;
	    tmp_cells_c3[i + j*nx] = tmp_1;
	    tmp_cells_c4[i + j*nx] = tmp_2;
	    tmp_cells_c5[i + j*nx] = tmp_7;
	    tmp_cells_c6[i + j*nx] = tmp_8;
	    tmp_cells_c7[i + j*nx] = tmp_5;
	    tmp_cells_c8[i + j*nx] = tmp_6;
	}

    const size_t local_id_x = get_local_id(0);
    const size_t local_id_y = get_local_id(1);
    const size_t local_size_x = get_local_size(0);
    const size_t local_size_y = get_local_size(1);
    int local_size = local_size_x * local_size_y;
    int local_id = local_id_x + local_size_x*local_id_y;

    partial_av_vels[local_id] = sqrt_u_sq;

	barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = local_size/2; i>0; i >>= 1) {
        if(local_id < i) {
            partial_av_vels[local_id] += partial_av_vels[local_id + i];
        }
		barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(local_id == 0){
        const size_t group_id_x = get_group_id(0); 
        const size_t group_id_y = get_group_id(1);
        const size_t number_groups_x = get_num_groups(0); 
        const size_t number_groups_y = get_num_groups(1);
        int group_id = group_id_x + number_groups_x * group_id_y;
        int number_local_groups = number_groups_x * number_groups_y;
        av_vels[group_id + number_local_groups*tt] = partial_av_vels[0];
    }
}
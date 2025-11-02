`timescale 1ns / 1ps

//module for nxn matrix multiplication
module MMU#(parameter depth=4, bit_width=8, acc_width=32,width=4)
(
    input clk,
    input control,
    input [(bit_width*depth)-1:0] data_arr,
    input [(bit_width*depth)-1:0] wt_arr,
    output wire [acc_width*width-1:0] acc_out_line
);

    //busses
    wire [bit_width-1:0] data_bus [0:width-1][0:width-1];  
    wire [bit_width-1:0] weight_bus [0:width-1][0:width-1]; 
    wire [acc_width-1:0] acc_bus [0:width-1][0:width-1];   
    
    wire [acc_width-1:0] pre_bus =0;

 
    reg [acc_width-1:0] final_reg [0:width-1];

    //nxn grid of MACs
    genvar i, j;
    generate
        for (i = 0; i < width; i = i + 1) begin : row_gen
            for (j = 0; j < width; j = j + 1) begin : col_gen
                MAC #(.bit_width(bit_width), .acc_width(acc_width)) mac_unit (
                    .clk(clk),
                    .control(control),
                    .acc_in((i==0)? pre_bus :acc_bus[i-1][j]),
                    .acc_out(acc_bus[i][j]),
                    .data_in((j == 0) ? data_arr[i*bit_width +: bit_width] : data_bus[i][j-1]),
                    .data_out(data_bus[i][j]),
                    .wt_path_in((i == 0) ? wt_arr[j*bit_width +: bit_width] : weight_bus[i-1][j]),
                    .wt_path_out(weight_bus[i][j])
                );
            end
        end
    endgenerate

    //final accumulation result
    always @(posedge clk) begin
        for (integer k = 0; k < width; k = k + 1) begin
            final_reg[k] <= acc_bus[width-1][k];
        end
    end

    //concatenation
    genvar k;
    generate
        for ( k = 0; k < width; k = k + 1) begin
            assign acc_out_line[k*acc_width+ acc_width - 1 : k*acc_width ] = final_reg[k];
        end
    endgenerate

endmodule

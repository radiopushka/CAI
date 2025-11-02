`timescale 1ns / 1ps

module MAC #(parameter bit_width=8, acc_width=32)(
    input clk,
    input control,          
    input [acc_width-1:0] acc_in,       
    output reg [acc_width-1:0] acc_out,
    input [bit_width-1:0] data_in,     
    output reg [bit_width-1:0] data_out, 
    input [bit_width-1:0] wt_path_in,  
    output reg [bit_width-1:0] wt_path_out
);

   
    reg [bit_width-1:0] data_reg;    
    reg [bit_width-1:0] wt_reg;        
    reg [acc_width-1:0] acc_reg;       

    wire [2*bit_width-1:0] product;  


    assign product = data_reg * wt_reg;


    always @(posedge clk) begin
      
        data_reg <= data_in;
        wt_reg <= wt_path_in;

  
        data_out <= data_reg;
        wt_path_out <= wt_reg;

  
        if (control == 1'b0) begin
         
            acc_reg <= acc_in + product;
        end else begin
         
            acc_reg <= acc_in;
        end

    
        acc_out <= acc_reg;
    end

endmodule
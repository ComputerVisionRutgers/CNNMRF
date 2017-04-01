--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
-- Created by: Hang Zhang
-- ECE Department, Rutgers University
-- Email: zhang.hang@rutgers.edu
-- Copyright (c) 2017
--
-- Free to reuse and distribute this software for research or 
-- non-profit purpose, subject to the following conditions:
--  1. The code must retain the above copyright notice, this list of
--     conditions.
--  2. Original authors' names are not deleted.
--  3. The authors' names are not used to endorse or promote products
--      derived from this software 
--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

require 'mylib.utils'

local FPLayer, parent = torch.class('nn.FacePriorLayer', 'nn.Module')

function FPLayer:__init(mask)
	parent.__init(self)
	self.mask = mask
end

function FPLayer:updateOutput(input)
	self.output = input:cmul(mask.add_dummy():expandAs(input))
	return self.output
end

function FPLayer:updateGradInput(input, gradOutput)
	self.gradInput = gradOutput:cmul(mask.add_dummy():expandAs(input))
	return self.gradInput
end

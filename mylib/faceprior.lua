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

require 'hzproc'
local FPLayer, parent = torch.class('nn.FacePriorLayer', 'nn.Module')

function FPLayer:__init(prior)
	parent.__init(self)
	self.mask = prior:clone()
end

function FPLayer:updateOutput(input)
	local HH, WW = input:size(2), input:size(3)
	local mH, mW = self.mask:size(1), self.mask:size(2)
	if HH ~= mH or WW ~= mW then
		if cutorch then
			local map = hzproc.Table.Resize(mW, mH, WW, HH)
			self.mask = hzproc.Remap.Bilinear(self.mask:add_dummy(), map):squeeze()
		else
			self.mask = image.scale(self.mask, WW, HH)
		end
	end
	self.output = input:cmul(self.mask:add_dummy():expandAs(input))
	return self.output
end

function FPLayer:updateGradInput(input, gradOutput)
	self.gradInput = gradOutput:cmul(self.mask.add_dummy():expandAs(input))
	return self.gradInput
end

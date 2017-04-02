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

require 'image'
local M = {}

-- load the binary prior image and do gaussian blur
function M:get(name, size, sigma)
	size = size or 30
	sigma = sigma or 0.5
	local img = image.load('data/face_prior/' .. name .. '_fp.png',1,'float')
	local kernel = image.gaussian(size, sigma):float()
	img = image.convolve(img, kernel, 'same')
	return 1-(img-img:min()):div(img:max()-img:min())
end

return M

--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "nn"
require "image"

local scale = torch.class('nn.Scale', 'nn.Module')


function scale:__init(width, height, focus_x, focus_y)
    self.height = height    -- height and width of the frame
    self.width = width

    self.focus_x = focus_x/2    -- focus width and height
    self.focus_y = focus_y/2
end

--[[
now converts image to
]]
function scale:forward(x, focus)
    local x = x
    x = image.rgb2y(x)    -- convert to single channel
    x = image.scale(x, self.width, self.height, 'bilinear')   -- compress to size if needed
    -- now crop out part in focus
    local focused = x[{ {} ,{math.floor(focus.y - self.focus_y) + 1, math.floor(focus.y + self.focus_y)} , {math.floor(focus.x - self.focus_x) + 1, math.floor(focus.x + self.focus_x)} }]
    return focused
end

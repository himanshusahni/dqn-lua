--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "image"
require "Scale"

local function create_network(args)
    -- Y (luminance)
    return nn.Scale(args.full_dims[2], args.full_dims[3], args.focus_x, args.focus_y)
end

return create_network

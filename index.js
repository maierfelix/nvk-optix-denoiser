const fs = require("fs");
const path = require("path");

let {platform} = process;

module.exports = require(path.join(__dirname, `build/Debug/addon-${platform}.node`));

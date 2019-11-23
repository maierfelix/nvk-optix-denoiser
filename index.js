const fs = require("fs");
const path = require("path");

let {platform} = process;

module.exports = require(path.join(__dirname, `build/Release/addon-${platform}.node`));

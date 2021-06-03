function removeBorderCells(cells){
    var i = 0;
    while(i < cells.length){
        if(cells[i][0][2] === -2)
            cells.splice(i, 1);
        else
            ++i;
    }
    return cells;
}

function mergeBoxCells(cells){
    var i = 0;
    while(i < cells.length){
        if(cells[i][0][2] >= 0){
            data_index = cells[i][0][2];
            if(cells[i][1])
                cells[data_index][1] = d3.polygonHull(cells[data_index][1].concat(cells[i][1]));
            cells.splice(i, 1);
        }
        else
            ++i;
    }
}

function removeSupportPoints(vertices){
    var i = 0;
    while(i < vertices.length){
        if(vertices[i][2] !== -1)
            vertices.splice(i, 1);
        else
            ++i;
    }
}


var BrowserText = (function () {
    var canvas = document.createElement('canvas'),
        context = canvas.getContext('2d');

    /**
     * Measures the rendered width of arbitrary text given the font size and font face
     * @param {string} text The text to measure
     * @param {number} fontSize The font size in pixels
     * @param {string} fontFace The font face ("Arial", "Helvetica", etc.)
     * @returns {number} The width of the text
     **/
    function getSize(text, fontSize, fontFace) {
        context.font = fontSize + 'px ' + fontFace;
        return context.measureText(text).width;
    }

    return {
        getSize: getSize
    };
})();


const  width = 1000, height = 600, displayThreshold = 2000, scale_factor = 600,
    font_height = 3, font_family = "sans-serif";
var c10 = d3.schemePaired;
//var vertices = d3.range(10).map(function(d) {
//  return [Math.random() * width, Math.random() * height];
//});
vertices.forEach(x => {
    x[0] = (x[0] * scale_factor / 2) + scale_factor / 2;
    x[1] = (x[1] * scale_factor / 2) + scale_factor / 2
})
//console.log(vertices);

data = vertices;
orient = ({
    top: text => text.attr("text-anchor", "middle").attr("y", -6),
    right: text => text.attr("text-anchor", "start").attr("dy", "0.35em").attr("x", 6),
    bottom: text => text.attr("text-anchor", "middle").attr("dy", "0.71em").attr("y", 6),
    left: text => text.attr("text-anchor", "end").attr("dy", "0.35em").attr("x", -6)
})


var svg = d3.select("body").append("svg").attr("viewBox", [0, 0, width, height]);//.attr("width", width).attr("height", height);
const g = svg.append("g");
var path = g.selectAll("path");


const delaunay = d3.Delaunay.from(data);
const voronoi = delaunay.voronoi([0, 0, width, height]);
const cells = data.map((d, i) => [d, voronoi.cellPolygon(i)]);


removeBorderCells(cells)
mergeBoxCells(cells)
removeSupportPoints(vertices)

path.data(cells).enter().append("path")
    .attr("stroke", "none")
    .attr("fill", function (d, i) {
        var country = d[0][3];
        return c10[country%10]
    })
    //    .attr("d", function(d) { return "M" + d.join("L") + "Z" } );
    .attr("d", polygon);

function polygon(b) {
    //console.log(b)
    d = b[1]
    //TODO why can this be null?
    if (b[0][2] !== -1 || d === null)
        return null;
    else
        return "M" + d.join("L") + "Z";
}

g.selectAll("circle").data(vertices).enter().append("circle").attr("r", 0.3)
    //.attr("transform", function(d) { return "translate(" + d + ")"; })
    .attr("cx", function (d) {
        if (d[2] !== -2 && draw_cirlce){
            return d[0];
        }
        else
            null;
    })
    .attr("cy", function (d) {
        if (d[2] !== -2 && draw_cirlce){
            return d[1];
        }
        else
            null;
    });


g.selectAll("line").data(edges).enter().append("line")
    .attr("x1", (d) => vertices[d[0]][0])
    .attr("y1", (d) => vertices[d[0]][1])
    .attr("x2", (d) => vertices[d[1]][0])
    .attr("y2", (d) => vertices[d[1]][1])
    .attr("style", "stroke:rgb(70,70,70);stroke-width:0.3;stroke-opacity: .8");

//console.log({{ tags | safe  }})
g.attr("class", "label")
    //.style("font", font_height + "px " + font_family)
    .selectAll("text")
    .data(cells)
    .join("text")
    .each(function ([[x, y], cell]) {
        //console.log(cell)
        //console.log(d3.polygonArea(cell));
        //cell.scaleThreshold = Math.min(Math.sqrt(displayThreshold / Math.abs(d3.polygonArea(cell))), 10);
        cell.scaleThreshold = 0
        cell.opacityScale = d3.scaleLinear().domain([cell.scaleThreshold, cell.scaleThreshold * 2]).range([1, 1]);
        //console.log(cell)
    })
    .attr("font-size",function ([, cell], i) {
        //cell.fs = Math.max(Math.floor(Math.log(Math.abs(d3.polygonArea(cell))))-2.5,1);
        cell.fs = 3;
        return cell.fs;
    })
    //.attr("transform", ([d]) => `translate(${d.slice(0, -1)})`)
    .attr("transform", function (d, i) {
        data_point = d[0];
        cell = d[1];
        const text = tag_names[i];
        if (typeof text !== 'undefined') {
            let lines = text.split(" ");
            let longest = 0
            for (const line in lines) {
                let current = BrowserText.getSize(lines[line], cell.fs, font_family)
                longest = ((current > longest) ? current : longest);
            }
            const position = data_point.slice(0, -1);
            //var position = d3.polygonCentroid(cell);
            position[0] -= longest / 2;
            position[1] -= ((cell.fs+1)/2) * lines.length;
            return `translate(${position[0]}, ${position[1]})`
        }
    }
    )
    //.attr("display", ([, cell]) => -d3.polygonArea(cell) > 2000 ? null : "none")
    .html(function (d, i) {
        //console.log(d)

        const text = tag_names[i];
        if (typeof text !== 'undefined') {
            let lines = text.split(" ");
            let longest = 0
            for (const line in lines) {
                let current = BrowserText.getSize(lines[line], d[1].fs, font_family)
                longest = ((current > longest) ? current : longest);
            }
            let str = "";
            for (const line in lines) {
                let current = BrowserText.getSize(lines[line], d[1].fs, font_family)
                str += `<tspan x='${((longest - current)/2)}px' dy='${d[1].fs}px'> ${lines[line]} </tspan>`;
            }
            return str;
        }
        else
            return ""
    })
    .attr('opacity', function (d) {
        return 1;
        if (d.scaleThreshold < 1) {
            return 1;
        }
        return 0;
    });


svg.call(d3.zoom()
    .extent([[0, 0], [width, height]])
    .scaleExtent([1, 24])
    .on("zoom", zoomed));

function zoomed({transform}) {
    g.attr("transform", transform);
    //g.attr("transform", "translate(" + d3.event.translate + ").scale(" + d3.event.scale + ")");
    //d3.selectAll('circle').attr('transform', function(d) {
    //  return 'translate(' + d + ')scale(' + (1/d3.event.scale) + ')';
    //});

    d3.selectAll('text')
        //.attr('transform', transform)
        .attr('opacity', function (d) {
            if (transform.k > d[1].scaleThreshold) {
                return d[1].opacityScale(transform.k);
            }
            return 0;
        });
}

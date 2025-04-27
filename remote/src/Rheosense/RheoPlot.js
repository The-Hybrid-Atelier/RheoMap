import {pulses} from "./RheoResults.js"

let _ = require('underscore');
let paper = require('paper');

const WIDTH = 300;
const HEIGHT = 100;
const SP_GAP = 10; // subplot gap
const BAR_GAP = 3; // subplot gap


function makeLinePlot(){
  console.log(pulses);

  const lp = new paper.Group({
    name: "lineplot"
  })
  const bg = paper.Path.Rectangle({
    parent: lp,
    name: "bg",
    rectangle: new paper.Size(paper.view.bounds.width, paper.view.bounds.height), 
    fillColor: "white",
    position: paper.view.center, 
    strokeColor: new paper.Color(0.3, 0.3, 0.3),
    strokeWidth: 0
  });


  let l = new paper.Path.Line({
      name: 'gridlines',
      from: bg.bounds.topCenter,
      to: bg.bounds.bottomCenter,
      strokeColor: new paper.Color(0, 0, 0, 0.3),
      strokeWidth:  2,
      opacity: 0.3, 
      parent: lp
    })
  let quarter = bg.bounds.topCenter.subtract(bg.bounds.topLeft).multiply(0.5)
  new paper.Path.Line({
    name: 'gridlines',
    from: quarter.add(bg.bounds.topCenter),
    to: quarter.add(bg.bounds.bottomCenter),
    strokeColor: new paper.Color(0, 0, 0, 0.3),
    strokeWidth:  1,
    opacity: 0.2, 
    parent: lp
  })
  
  new paper.Path.Line({
    name: 'gridlines',
    from: quarter.add(bg.bounds.topLeft),
    to: quarter.add(bg.bounds.bottomLeft),
    strokeColor: new paper.Color(0, 0, 0, 0.3),
    strokeWidth:  1,
    opacity: 0.2, 
    parent: lp
  })

  new paper.Path.Line({
      name: 'gridlines',
      from: bg.bounds.leftCenter,
      to: bg.bounds.rightCenter,
      strokeColor: new paper.Color(0, 0, 0, 0.3),
      strokeWidth:  2,
      opacity: 0.6, 
      parent: lp
    })
  quarter = bg.bounds.leftCenter.subtract(bg.bounds.topLeft).multiply(0.5)
  new paper.Path.Line({
    name: 'gridlines',
    from: quarter.add(bg.bounds.leftCenter),
    to: quarter.add(bg.bounds.rightCenter),
    strokeColor: new paper.Color(0, 0, 0, 0.3),
    strokeWidth:  1,
    opacity: 0.2, 
    parent: lp
  })
  
  new paper.Path.Line({
    name: 'gridlines',
    from: quarter.add(bg.bounds.topRight),
    to: quarter.add(bg.bounds.topLeft),
    strokeColor: new paper.Color(0, 0, 0, 0.3),
    strokeWidth:  1,
    opacity: 0.2, 
    parent: lp
  })  
}

function plotProfile(pulses, pulseColor){

  let color = new paper.Color(pulseColor)
  let mean_color = color.clone()
  let interval_fill = color.clone()
  let interval_boundary = color.clone()

  

  mean_color.alpha = 0.7;
  // mean_color.darkness = 0.3;
  interval_fill.alpha = 0.4;
  interval_boundary.alpha = 0.3;
  const INTERVAL_STROKE_WIDTH = 2;
  const MEAN_STROKE_WIDTH = 3;

  let lp = paper.project.getItem({name: "lineplot"})
  let bg = paper.project.getItem({name: "bg"})
  let lines = new paper.Group({
    name: "lines", 
    parent: lp
  });

  // HANDLE THE NAN CASE
  let a = _.map(pulses.interval_ub, function(pt, i){ return _.isNaN(pt) ? new paper.Point(i, pulses.mean[i]) : new paper.Point(i, pt); })
  let b = _.map(pulses.interval_lb, function(pt, i){ return _.isNaN(pt) ? new paper.Point(i, pulses.mean[i]) : new paper.Point(i, pt); })
  b = b.reverse()

  console.log(pulses.mean)

  let segments = _.flatten([a, b]);

  let interval = new paper.Path({
    name: 'interval68', 
    strokeColor: interval_boundary,
    strokeWidth:  INTERVAL_STROKE_WIDTH,
    parent: lines, 
    segments: segments,
    fillColor: interval_fill,
    closed: true
  })
  

  let ml = new paper.Path({
      name: 'mean',
      segments: _.map(pulses.mean, function(pt, i){ return new paper.Point(i, pt); }),
      strokeColor: mean_color,
      strokeWidth:  MEAN_STROKE_WIDTH,
      parent: lines, 
  });



  lines.fitBounds(bg.bounds)
  let scalex = bg.bounds.width/lines.bounds.width
  let scaley = -(bg.bounds.height - 100)/lines.bounds.height // correct y orientation, leave small gap above and below
  lines.scaling.x = scalex;
  lines.scaling.y = scaley;
}

function makeBarPlot(){
  const BG_STROKE = 0;
  const bar = new paper.Group({
    name: "bar"
  })
  const pc1 = new paper.Group({
    parent: bar,
    name: "component",
    pc: 1,
  })
  const pc1_bg = paper.Path.Rectangle({
    parent: pc1,
    name: "bg",
    rectangle: new paper.Size(WIDTH, HEIGHT), 
    fillColor: "white",
    position: paper.view.center, 
    strokeColor: new paper.Color(0.3, 0.3, 0.3),
    strokeWidth: BG_STROKE
  });
  
  const pc2 = new paper.Group({
    parent: bar,
    name: "component",
    pc: 2,
  })
  const pc2_bg = paper.Path.Rectangle({
    parent: pc2,
    name: "bg",
    rectangle: new paper.Size(WIDTH, HEIGHT), 
    fillColor: "white",
    strokeColor: new paper.Color(0.3, 0.3, 0.3),
    strokeWidth: BG_STROKE,
    position: pc1.bounds.center.add(new paper.Point(0, pc1.bounds.height + SP_GAP))
  });
  const pc3 = new paper.Group({
    parent: bar,
    name: "component",
    pc: 3,
  })
  const pc3_bg = paper.Path.Rectangle({
    parent: pc3,
    name: "bg",
    rectangle: new paper.Size(WIDTH, HEIGHT), 
    fillColor: "white",
    strokeColor: new paper.Color(0.3, 0.3, 0.3),
    strokeWidth: BG_STROKE,
    position: pc2.bounds.center.add(new paper.Point(0, pc2.bounds.height + SP_GAP))
  });


  // text.position = 

  _.each([pc1, pc2, pc3], function(el, i){
    let l = new paper.Path.Line({
      name: 'gridlines',
      from: el.bounds.topCenter,
      to: el.bounds.bottomCenter,
      strokeColor: new paper.Color(0, 0, 0, 0.3),
      strokeWidth:  2,
      opacity: 0.8, 
      parent: el
    })
    let quarter = el.bounds.topCenter.subtract(el.bounds.topLeft).multiply(0.5)
    new paper.Path.Line({
      name: 'gridlines',
      from: quarter.add(el.bounds.topCenter),
      to: quarter.add(el.bounds.bottomCenter),
      strokeColor: new paper.Color(0, 0, 0, 0.3),
      strokeWidth:  1,
      opacity: 0.4, 
      parent: el
    })
    
    new paper.Path.Line({
      name: 'gridlines',
      from: quarter.add(el.bounds.topLeft),
      to: quarter.add(el.bounds.bottomLeft),
      strokeColor: new paper.Color(0, 0, 0, 0.3),
      strokeWidth:  1,
      opacity: 0.4, 
      parent: el
    })

    let text = new paper.PointText({
      content: 'PC' + (i+1),
      fillColor: 'black',
      fontFamily: 'Avenir',
      // fontWeight: 'bold',
      fontSize: 25,
      parent: bar
    });
    
    text.pivot = text.bounds.rightCenter.add(new paper.Point(SP_GAP, 0));
    text.position = el.bounds.leftCenter;

  })
  // let text = new paper.PointText({
  //   content: title,
  //   fillColor: 'black',
  //   fontFamily: 'Roboto',
  //   fontWeight: 'bold',
  //   fontSize: 25,
  //   parent: bar
  // });
  // let components = paper.project.getItems({name: "component"});
  // text.pivot = text.bounds.bottomCenter.add(new paper.Point(0, SP_GAP));
  // text.position = components[0].bounds.topCenter;

  bar.position = paper.view.center
}

function makeBarProfile(data, color){
  let initValues = data.pca
  let numComponents = initValues.length;
  let components = paper.project.getItems({name: "component"});
  // console.log(initValues)
  _.each(components, function(pc, i){
    let wrapper = pc.getItem({name: "wrapper"})

    if(!wrapper){
      wrapper = new paper.Group({
        parent: pc,
        name: "wrapper",
        position: pc.bounds.topLeft, 
        applyMatrix: true
      })
    }
    
    let current = pc.getItems({name: "d"});
    let a = paper.Path.Rectangle({
      name: "d",
      rectangle: new paper.Size(pc.bounds.width * initValues[i], HEIGHT/3), 
      fillColor: color,
      position: new paper.Point(0, 0)
    });
    a.pivot = a.bounds.topLeft;
    if(current.length == 0){
      a.position = pc.bounds.topLeft
    }else{
      // console.log("HERE", current.slice(-1))
      a.position = current.slice(-1)[0].bounds.bottomLeft.add(new paper.Point(0, BAR_GAP))
      // a.position = paper.view.center
    }
    a.parent = wrapper
    // _.each(pc.getItems({name: "gridlines"}), function(el){ el.bringToFront(); });
    
    // console.log()
  })

}
function resizePlot(){
  let components = paper.project.getItems({name: "component"});
  _.each(components, function(pc, i){
    let bg = pc.getItem({name: "bg"});
    let wrapper = pc.getItem({name: "wrapper"})
    if(bg && wrapper){
      wrapper.scaling = new paper.Point(1, (bg.bounds.height-2* BAR_GAP)/wrapper.bounds.height)
      wrapper.pivot = wrapper.bounds.topLeft
      wrapper.position = bg.bounds.topLeft.add(new paper.Point(0 , BAR_GAP))
    }
  });
}



function makePlot(numComponents){
    const bg = paper.Path.Circle({
        radius: 100, 
        fillColor: "white",
        position: paper.view.center
    });
  
    _.each([0, 25, 50, 75, 100], (el, i)=>{
      paper.Path.Circle({
        radius: el, 
        strokeColor: "black",
        strokeWidth: i % 2 === 1 ? 1 : 2,
        opacity: i % 2 === 1 ? 0.5 : 1.0,
        position: paper.view.center
      });
      // console.log( i % 2 === 1 ? 0.5 : 1.0, i, el);
    });
  
    const star = new paper.Path.Star({
      points: numComponents, 
      radius1: 10, 
      radius2: 50, 
      center: paper.view.center
    });
    star.pivot = paper.view.center
    star.rotate(-120, paper.view.center);
   
    star.components = _.map(_.range(0, numComponents), (el, i)=>{
      return star.segments[el * 2 + 1]
    });
    star.axes = _.map(star.components, (el, i)=>{
      let pt = el.point;
      let axis =  pt.subtract(paper.view.center);
      axis.length = 100;
      return axis;
    });
    
    star.labels = _.map(star.axes, (axis, i)=>{
      let text = new paper.PointText({
        point: paper.view.center,
        content: 'PC' + i,
        fillColor: 'black',
        fontFamily: 'Roboto',
        fontWeight: 'normal',
        fontSize: 25,
      });
      axis.length = 135;
      text.position = axis.add(paper.view.center)
      return text;
    });
  
    star.remove()
  }
  

  
function makeProfile(numComponents, color, initValues){
    let clearColor = new paper.Color(color)
    clearColor.alpha = 0.2;
    const origami = new paper.Group({
      name: "origami"
    });
    const star = new paper.Path.Star({
      parent: origami,
      points: numComponents, 
      radius1: 10, 
      radius2: 50, 
      fillColor: clearColor, 
      opacity: 1.0,
      strokeColor: darken(color, 0.20),
      strokeWidth: 2,
      miterLimit: 1,
      center: paper.view.center
    });
    origami.star = star;
  
    star.pivot = paper.view.center
    star.rotate(-120, paper.view.center);
   
    star.components = _.map(_.range(0, numComponents), (el, i)=>{
      return star.segments[el * 2 + 1]
    });
  
    star.cpoints = _.map(star.components, (el)=>{
      const pt = paper.Path.Circle({
        radius: 4, 
        fillColor: darken(color, 0.30),
        position: el.point,
        parent: origami
      });
      return pt;
    })
  
    star.axes = _.map(star.components, (el, i)=>{
      let pt = el.point;
      let axis =  pt.subtract(paper.view.center);
      axis.length = 100;
      return axis;
    });
    
    star.setComponent = function(id, value){
      this.axes[id].length = value * 100;
      this.components[id].point = paper.view.center.add(this.axes[id]);
      this.cpoints[id].position = paper.view.center.add(this.axes[id]);
    }
    origami.setComponents = function(arr){
      let that = this;
      _.each(arr, function(el, i){
        that.star.setComponent(i, el);
      });
    }
    origami.setComponents(initValues);
    return origami;
}


function closestPivot(object, refPt){
    let pivots = ["topLeft", "topRight", "bottomLeft", "bottomRight", "leftCenter", "topCenter", "rightCenter", "bottomCenter"]
    return _.min(pivots, (pt)=>{
      return object.bounds[pt].getDistance(refPt, true);
    });
}

function darken(c, amount){
    let color = new paper.Color(c);
    color.lightness -= amount;
    return color;
}

export {makePlot, makeProfile, makeBarPlot, makeLinePlot, plotProfile, makeBarProfile, resizePlot};
let paper = require('paper');
let FileSaver = require('file-saver');

function setupPaper(canvasRef){
  // explicitly set the attr width/height
  canvasRef.current.height = canvasRef.current.parentElement.height
  canvasRef.current.width = canvasRef.current.parentElement.width
  paper.setup(canvasRef.current);
  paper.view.zoom = 1
  // console.log(paper.view.size);
}
function exportSVG(){
  let prev = paper.view.zoom;
  console.log("Exporting file as SVG");
  paper.view.zoom = 1;
  paper.view.update();
  let exp = paper.project.exportSVG({
    asString: true,
    precision: 5
  })
  FileSaver.saveAs(new Blob([exp], {type:"application/svg+xml"}), "test.svg")
  paper.view.zoom = prev
}

export {setupPaper, exportSVG};
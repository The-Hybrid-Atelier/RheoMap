import {Row, Column, Pill} from './Components.js';

function View({children, icon, screen, goToScreen, setScreen, padding}){
  const handleCancel = (event)=>{setScreen(goToScreen)}

  return (
    <Column className={`view ${padding} ${screen}`}>
      <Row className="right paddedlr">
        <button class="primary-action">
          <img width={40} height={40} onClick={handleCancel} src={`/${icon}.png`} size='small' />
        </button>
      </Row>
      {children}
    </Column>
  );
}
// const ActionPicker = function({screen, setScreen, exportSVG}){
//   return (
//     <div id="view-picker">
//       <Button.Group>
//         { ["pulse", "library"].includes(screen) &&
//         <Button icon onClick={()=> setScreen("library")}>
//           <img src="/plus.png"/>
//         </Button>
//         }
//         { false &&
//           <Button icon onClick={exportSVG}>
//             <Icon name='download'  />
//           </Button>
//         }
//       </Button.Group> 
//     </div>
//   );
// }

export {View};
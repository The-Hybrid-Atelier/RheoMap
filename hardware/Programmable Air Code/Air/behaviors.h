void inflate()
{

  switchOnPump(2, 60);
  switchOffPump(1);
  blow();
  delay(2000);
  switchOnPump(2, 30);
}

void deflate()
{
  switchOnPump(1, 100);
  setAllValves(OPEN);
  delay(3000);
  // switchOffPumps();
}

void pulsing()
{
  inflate();
  delay(500);
  deflate();
  delay(500);
}
/// @brief Clear out the PA system
void clear_out()
{
  switchOnPump(2, 100);
  switchOffPump(1);
  blow();

  delay(1000);
  switchOffPumps();
}

///// RheoPulse Routine ////////
void rheosense_routine()
{
  delay(250);
  switchOnPump(1, 100);
  switchOffPump(2);
  suck();
  delay(250);
  switchOnPump(2, 100);
  switchOffPump(1);
  blow();
  delay(250);
  switchOffPumps();
}

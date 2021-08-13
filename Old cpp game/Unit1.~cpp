//---------------------------------------------------------------------------

#include <vcl.h>
#pragma hdrstop

#include "Unit1.h"
//---------------------------------------------------------------------------
#pragma package(smart_init)
#pragma resource "*.dfm"
TGame *Game;
int m,sec;
int s = 0;
int score = 0;
//TImage* a[9];

int vy = 3; int vx = 3;

class ImgVelocity{
        public:
        TImage* alk;
        int vx,vy;
        ImgVelocity(TImage* a, int x, int y){
                alk = a;
                alk->Top = (std::rand() % (Game->Height));
                alk->Left = (std::rand() % (Game->Width));
                vx = x;
                vy = y;
        }
} ;


ImgVelocity* iv[9];

void move(ImgVelocity *iv){
        //odbicie od prawej sciany
        if(iv->alk->Left + iv->alk->Width >= Game->Width) iv->vx = -iv->vx;
        //odbicie od lewej sciany
        if(iv->alk->Left  <= 0) iv->vx = -iv->vx;
        //odbicie od dolnej sciany
        if(iv->alk->Top + iv->alk->Height >= Game->Height - iv->alk->Height) iv->vy = -iv->vy;
        //odbicie od gornej sciany
        if(iv->alk->Top <= 0) iv->vy = -iv->vy;

        iv->alk->Left += iv->vx;
        iv->alk->Top += iv->vy;
}

//---------------------------------------------------------------------------

void scorePoint(ImgVelocity *iv, TTimer *movement, TLabel *scoreL){
     if(movement -> Enabled == true){
                iv->alk -> Visible = false;
                score++;
                scoreL -> Caption = "Score: "+ IntToStr(score);
        }
}

void reset(){
        Game->Movement -> Enabled = false;
        Game->Timer -> Enabled = false;
        Game->Time -> Visible = false;
        Game->Score -> Visible = false;
        Game->Begin -> Visible = true;
        score = 0; s = 0;
        Game->Score -> Caption = "Score: "+ IntToStr(score);
        Game->Time -> Caption = "Time: ";
        Game->Win->Visible = false;
        Game->Reset->Visible = false;
        Game->Description->Visible = true;

        for(int i = 0; i<9; i++){
                iv[i]->alk->Visible = false;
                iv[i]->alk->Top = (std::rand() % (Game->Height));
                iv[i]->alk->Left = (std::rand() % (Game->Width));
            }
}

void changeVelocity(ImgVelocity *iv, int x, int y){
        if(iv->vx<0)
                iv->vx = -x;
        else
                iv->vx = x;
        if(iv->vy<0)
                iv->vy = -y;
        else
                iv->vy = y;

}

void changeDifficulty(int v){
        for(int i = 0 ; i<9; i++)
                changeVelocity(iv[i],v,v);
}

//---------------------------------------------------------------------------
__fastcall TGame::TGame(TComponent* Owner)
        : TForm(Owner)
{
        //a[0] = alkSuper1; a[1] = alkSuper2; a[2] = alkSuper3;
        //a[3] = alkSuper4; a[4] = alkSuper5; a[5] = alkSuper6;
        //a[6] = alkSuper7; a[7] = alkSuper8; a[8] = alkSuper9;
        iv[0] = new ImgVelocity(alkSuper1,vx,vy); iv[1] = new ImgVelocity(alkSuper2,vx,vy);
        iv[2] = new ImgVelocity(alkSuper3,vx,vy); iv[3] = new ImgVelocity(alkSuper4,vx,vy);
        iv[4] = new ImgVelocity(alkSuper5,vx,vy); iv[5] = new ImgVelocity(alkSuper6,vx,vy);
        iv[6] = new ImgVelocity(alkSuper7,vx,vy); iv[7] = new ImgVelocity(alkSuper8,vx,vy);
        iv[8] = new ImgVelocity(alkSuper9,vx,vy);
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------



void __fastcall TGame::Exit1Click(TObject *Sender)
{

    if(Application->MessageBox(
    "Czy na pewno zakoñczyæ grê?","PotwierdŸ",
    MB_YESNO | MB_ICONQUESTION) == IDYES )
    {
        Application->Terminate();
    }
}
//---------------------------------------------------------------------------

void __fastcall TGame::FormClose(TObject *Sender, TCloseAction &Action)
{

    if(Application->MessageBox(
    "Czy na pewno zakoñczyæ grê?","PotwierdŸ",
    MB_YESNO | MB_ICONQUESTION) == IDNO )
    {
        Action = caNone;
    }
}
//---------------------------------------------------------------------------

void __fastcall TGame::Author1Click(TObject *Sender)
{
        ShellExecute(NULL,"open","http://math.uni.wroc.pl/~s297759", NULL, NULL, SW_SHOWNORMAL);
}
//---------------------------------------------------------------------------

void __fastcall TGame::Project1Click(TObject *Sender)
{
        ShellExecute(NULL,"open","http://math.uni.wroc.pl/~s297759", NULL, NULL, SW_SHOWNORMAL);        
}
//---------------------------------------------------------------------------



void __fastcall TGame::TimerTimer(TObject *Sender)
{
        AnsiString sM, sSec;
         s++;
         m =  s /60;
         sec = s - (m * 60);

         sM = IntToStr(m);
         if(m<10) sM = "0"+sM;
         sSec = IntToStr(sec);
         if(sec<10) sSec = "0"+sSec;

         Time -> Caption = "Time: "+sM + ":" + sSec;
         
}
//---------------------------------------------------------------------------


void __fastcall TGame::MovementTimer(TObject *Sender)
{
        for(int i = 0; i<9; i++){
                //move(a[i]);
                move(iv[i]);
        }
}
//---------------------------------------------------------------------------


void __fastcall TGame::BeginClick(TObject *Sender)
{
            Movement -> Enabled = true;
            Timer -> Enabled = true;
            Time -> Visible = true;
            Score -> Visible = true;
            Begin -> Visible = false;
            Description->Visible = false;

            for(int i = 0; i<9; i++){
                iv[i]->alk->Visible = true;
            }
}

//---------------------------------------------------------------------------

void __fastcall TGame::Pause1Click(TObject *Sender)
{
        Movement -> Enabled = false;
        Timer -> Enabled = false;
}
//---------------------------------------------------------------------------

void __fastcall TGame::Resume1Click(TObject *Sender)
{
         Movement -> Enabled = true;
         Timer -> Enabled = true;
}
//---------------------------------------------------------------------------

void __fastcall TGame::Reset1Click(TObject *Sender)
{
      reset();
}
//---------------------------------------------------------------------------
void __fastcall TGame::ResetClick(TObject *Sender)
{
    reset();
}

//---------------------------------------------------------------------------

void __fastcall TGame::alkSuper1Click(TObject *Sender)
{
        scorePoint(iv[0], Movement, Score);
        if(score == 9){
                Win->Visible = true;
                Reset->Visible = true;
                Timer->Enabled = false;
        }
}

void __fastcall TGame::alkSuper2Click(TObject *Sender)
{
     scorePoint(iv[1], Movement, Score);
     if(score == 9){
                Win->Visible = true;
                Reset->Visible = true;
                Timer->Enabled = false;
        }
}
//---------------------------------------------------------------------------

void __fastcall TGame::alkSuper3Click(TObject *Sender)
{
        scorePoint(iv[2], Movement, Score);
        if(score == 9){
                Win->Visible = true;
                Reset->Visible = true;
                Timer->Enabled = false;
        }
}
//---------------------------------------------------------------------------

void __fastcall TGame::alkSuper4Click(TObject *Sender)
{
        scorePoint(iv[3], Movement, Score);
        if(score == 9){
                Win->Visible = true;
                Reset->Visible = true;
                Timer->Enabled = false;
        }
}
//---------------------------------------------------------------------------

void __fastcall TGame::alkSuper5Click(TObject *Sender)
{
        scorePoint(iv[4], Movement, Score);
        if(score == 9){
                Win->Visible = true;
                Reset->Visible = true;
                Timer->Enabled = false;
        }
}
//---------------------------------------------------------------------------

void __fastcall TGame::alkSuper6Click(TObject *Sender)
{
        scorePoint(iv[5], Movement, Score);
        if(score == 9)
        if(score == 9){
                Win->Visible = true;
                Reset->Visible = true;
                Timer->Enabled = false;
        }
}
//---------------------------------------------------------------------------

void __fastcall TGame::alkSuper7Click(TObject *Sender)
{
        scorePoint(iv[6], Movement, Score);
        if(score == 9){
                Win->Visible = true;
                Reset->Visible = true;
                Timer->Enabled = false;
        }
}
//---------------------------------------------------------------------------

void __fastcall TGame::alkSuper8Click(TObject *Sender)
{
        scorePoint(iv[7], Movement, Score);
        if(score == 9){
                Win->Visible = true;
                Reset->Visible = true;
                Timer->Enabled = false;
        }
}
//---------------------------------------------------------------------------

void __fastcall TGame::alkSuper9Click(TObject *Sender)
{
        scorePoint(iv[8], Movement, Score);
        if(score == 9){
                Win->Visible = true;
                Reset->Visible = true;
                Timer->Enabled = false;
        }
}
//---------------------------------------------------------------------------





void __fastcall TGame::Easy1Click(TObject *Sender)
{
        vx = vy = 3;
        changeDifficulty(vx);
}
//---------------------------------------------------------------------------

void __fastcall TGame::MEdium1Click(TObject *Sender)
{
       vx = vy = 5;
       changeDifficulty(vx);
}
//---------------------------------------------------------------------------

void __fastcall TGame::Hard1Click(TObject *Sender)
{
        vx = vy = 7;
        changeDifficulty(vx);
}
//---------------------------------------------------------------------------


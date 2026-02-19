#property strict
#property description "R10 Professional EA (MT5) - conversao do roboMamhedgeR10.py"
#property version   "1.00"

#include <Trade/Trade.mqh>

input long   InpMagicNumber           = 10010;
input double InpLots                  = 1.0;

// Parametros do R10 (defaults do Python)
input int    InpEmaFast               = 10;
input int    InpEmaSlow               = 34;
input int    InpRsiPeriod             = 14;
input double InpRsiThresh             = 50.0;
input int    InpRsiWindow             = 3;
input double InpStopATR               = 2.0;
input double InpTrailATR              = 2.2;
input double InpBreakevenTriggerATR   = 2.2;
input bool   InpUseADX                = true;
input double InpADXMin                = 20.0;
input bool   InpUseMACD               = true;
input double InpMinATR                = 1e-9;
input int    InpMaxBarsInTrade        = 12;

// R10 foi desenhado para M5.
input ENUM_TIMEFRAMES InpTimeframe    = PERIOD_M5;

// Ajuste para converter hora do servidor para BRT.
// Exemplo: servidor UTC => -3. Servidor ja em BRT => 0.
input int    InpServerToBRTHours      = 0;

input bool   InpVerboseLogs           = true;

CTrade g_trade;

int g_hEmaFast = INVALID_HANDLE;
int g_hEmaSlow = INVALID_HANDLE;
int g_hRsi     = INVALID_HANDLE;
int g_hAtr     = INVALID_HANDLE;
int g_hAdx     = INVALID_HANDLE;
int g_hMacd    = INVALID_HANDLE;

datetime g_lastBarTime = 0;

bool   g_inPosition = false;
double g_entryPrice = 0.0;
double g_stopLoss = 0.0;
double g_highestSinceEntry = 0.0;
int    g_barsInTrade = 0;
int    g_entryDateKeyBRT = 0;

datetime ToBRT(datetime serverTime)
{
   return serverTime + (InpServerToBRTHours * 3600);
}

int DateKeyBRT(datetime serverTime)
{
   MqlDateTime dt;
   TimeToStruct(ToBRT(serverTime), dt);
   return (dt.year * 10000 + dt.mon * 100 + dt.day);
}

bool IsInSessionBRT(datetime serverTime)
{
   MqlDateTime dt;
   TimeToStruct(ToBRT(serverTime), dt);

   int h = dt.hour;
   int m = dt.min;

   // Janela: 10:45 (inclusive) ate 16:30 (exclusive), em BRT.
   if(h < 10 || h >= 17)
      return false;
   if(h == 10 && m < 45)
      return false;
   if(h == 16 && m >= 30)
      return false;
   return true;
}

bool CopyValue(int handle, int buffer, int shift, double &outVal)
{
   double buf[];
   int copied = CopyBuffer(handle, buffer, shift, 1, buf);
   if(copied != 1)
      return false;
   outVal = buf[0];
   return true;
}

bool PositionIsOurBuy()
{
   if(!PositionSelect(_Symbol))
      return false;

   long type = PositionGetInteger(POSITION_TYPE);
   long magic = PositionGetInteger(POSITION_MAGIC);

   if(type != POSITION_TYPE_BUY)
      return false;
   if(magic != InpMagicNumber)
      return false;

   return true;
}

void ResetState()
{
   g_inPosition = false;
   g_entryPrice = 0.0;
   g_stopLoss = 0.0;
   g_highestSinceEntry = 0.0;
   g_barsInTrade = 0;
   g_entryDateKeyBRT = 0;
}

void RecoverStateFromOpenPosition()
{
   if(!PositionIsOurBuy())
   {
      ResetState();
      return;
   }

   g_inPosition = true;
   g_entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
   double sl = PositionGetDouble(POSITION_SL);
   g_stopLoss = (sl > 0.0 ? sl : g_entryPrice);
   g_highestSinceEntry = g_entryPrice;
   g_barsInTrade = 0;

   datetime openTime = (datetime)PositionGetInteger(POSITION_TIME);
   g_entryDateKeyBRT = DateKeyBRT(openTime);
}

void LogMsg(string msg)
{
   if(InpVerboseLogs)
      Print("[R10_MT5_EA] ", msg);
}

bool UpdateBrokerStop(double newSL)
{
   if(!PositionIsOurBuy())
      return false;

   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double stopsLevelPts = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double minDistance = stopsLevelPts * _Point;

   double sl = newSL;
   if((bid - sl) < minDistance)
      sl = bid - minDistance;

   sl = NormalizeDouble(sl, _Digits);
   if(sl <= 0.0)
      return false;

   double currentSL = PositionGetDouble(POSITION_SL);
   if(currentSL > 0.0 && sl <= currentSL + (_Point * 0.1))
      return true; // nao piora stop

   bool ok = g_trade.PositionModify(_Symbol, sl, 0.0);
   if(!ok)
      LogMsg("Falha ao atualizar SL. retcode=" + IntegerToString((int)g_trade.ResultRetcode()));
   return ok;
}

bool CloseOurPosition(string reason)
{
   if(!PositionIsOurBuy())
   {
      ResetState();
      return false;
   }

   bool ok = g_trade.PositionClose(_Symbol);
   if(ok)
      LogMsg("Posicao fechada: " + reason);
   else
      LogMsg("Falha ao fechar posicao (" + reason + "). retcode=" + IntegerToString((int)g_trade.ResultRetcode()));

   ResetState();
   return ok;
}

bool RsiBullishWindow()
{
   // Equivalente de: (rsi > thresh).rolling(window=rsi_window).max()
   // usando barras fechadas: shifts 1..InpRsiWindow.
   bool anyBull = false;
   for(int s = 1; s <= InpRsiWindow; s++)
   {
      double rsi;
      if(!CopyValue(g_hRsi, 0, s, rsi))
         return false;
      if(!MathIsValidNumber(rsi))
         return false;
      if(rsi > InpRsiThresh)
         anyBull = true;
   }
   return anyBull;
}

bool OpenBuy(double atrVal, int dateKeyBRT)
{
   if(PositionIsOurBuy())
      return false;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(ask <= 0.0)
      return false;

   double initialSL = NormalizeDouble(ask - (InpStopATR * atrVal), _Digits);

   bool ok = g_trade.Buy(InpLots, _Symbol, 0.0, initialSL, 0.0, "R10 Pro EA");
   if(!ok)
   {
      LogMsg("Falha no BUY. retcode=" + IntegerToString((int)g_trade.ResultRetcode()));
      return false;
   }

   if(!PositionIsOurBuy())
      return false;

   g_inPosition = true;
   g_entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
   g_stopLoss = initialSL;
   g_highestSinceEntry = g_entryPrice;
   g_barsInTrade = 0;
   g_entryDateKeyBRT = dateKeyBRT;

   LogMsg("BUY aberto. entry=" + DoubleToString(g_entryPrice, _Digits) +
          " | stop=" + DoubleToString(g_stopLoss, _Digits));
   return true;
}

void ProcessClosedBar()
{
   if(Bars(_Symbol, InpTimeframe) < 200)
      return;

   // Leitura da ultima barra fechada (shift=1) e anterior (shift=2)
   datetime ts1 = iTime(_Symbol, InpTimeframe, 1);
   if(ts1 <= 0)
      return;

   double close1 = iClose(_Symbol, InpTimeframe, 1);
   double high1  = iHigh(_Symbol, InpTimeframe, 1);
   double low1   = iLow(_Symbol, InpTimeframe, 1);
   if(!MathIsValidNumber(close1) || !MathIsValidNumber(high1) || !MathIsValidNumber(low1))
      return;

   double emaFast1, emaFast2, emaSlow1, atr1;
   if(!CopyValue(g_hEmaFast, 0, 1, emaFast1)) return;
   if(!CopyValue(g_hEmaFast, 0, 2, emaFast2)) return;
   if(!CopyValue(g_hEmaSlow, 0, 1, emaSlow1)) return;
   if(!CopyValue(g_hAtr, 0, 1, atr1)) return;

   if(!MathIsValidNumber(emaFast1) || !MathIsValidNumber(emaFast2) || !MathIsValidNumber(emaSlow1))
      return;

   bool emaUp = (emaFast1 > emaFast2);
   bool regimeUp = (emaFast1 > emaSlow1);
   bool inSession = IsInSessionBRT(ts1);
   int dateKeyBRT = DateKeyBRT(ts1);

   // Sincroniza estado caso a plataforma reinicie com posicao aberta.
   bool hasPos = PositionIsOurBuy();
   if(hasPos && !g_inPosition)
      RecoverStateFromOpenPosition();
   if(!hasPos && g_inPosition)
      ResetState();

   // Gestao de posicao aberta (long-only)
   if(g_inPosition && hasPos)
   {
      g_barsInTrade++;

      // 0) Saida operacional obrigatoria
      if(dateKeyBRT != g_entryDateKeyBRT || !inSession)
      {
         CloseOurPosition("saida operacional");
         return;
      }

      // 1) Stop vigente (checa low da barra fechada vs stop calculado na barra anterior)
      if(low1 <= g_stopLoss)
      {
         CloseOurPosition("stop atingido");
         return;
      }

      // 2) Saida por regime ou time-stop
      if((!regimeUp) || (g_barsInTrade >= InpMaxBarsInTrade))
      {
         CloseOurPosition("regime/time-stop");
         return;
      }

      // 3) Atualiza trailing para vigorar nas barras seguintes
      g_highestSinceEntry = MathMax(g_highestSinceEntry, high1);

      if((g_highestSinceEntry - g_entryPrice) >= (InpBreakevenTriggerATR * atr1))
         g_stopLoss = MathMax(g_stopLoss, g_entryPrice);

      double trailingStop = g_highestSinceEntry - (InpTrailATR * atr1);
      g_stopLoss = MathMax(g_stopLoss, trailingStop);
      g_stopLoss = NormalizeDouble(g_stopLoss, _Digits);

      UpdateBrokerStop(g_stopLoss);
      return;
   }

   // Entrada
   if(!inSession)
      return;
   if(!regimeUp || !emaUp)
      return;

   if(InpUseADX)
   {
      double adx1;
      if(!CopyValue(g_hAdx, 0, 1, adx1))
         return;
      if(MathIsValidNumber(adx1) && adx1 < InpADXMin)
         return;
   }

   if(InpUseMACD)
   {
      double macdMain, macdSignal;
      if(!CopyValue(g_hMacd, 0, 1, macdMain))
         return;
      if(!CopyValue(g_hMacd, 1, 1, macdSignal))
         return;
      double macdHist = macdMain - macdSignal;
      if(MathIsValidNumber(macdHist) && macdHist <= 0.0)
         return;
   }

   if(!MathIsValidNumber(atr1) || atr1 <= InpMinATR)
      return;

   if(!RsiBullishWindow())
      return;

   OpenBuy(atr1, dateKeyBRT);
}

int OnInit()
{
   g_trade.SetExpertMagicNumber(InpMagicNumber);

   if(_Period != InpTimeframe)
      Print("[R10_MT5_EA] Aviso: grafico em timeframe diferente do InpTimeframe.");
   if(InpTimeframe != PERIOD_M5)
      Print("[R10_MT5_EA] Aviso: R10 foi calibrado para M5.");

   g_hEmaFast = iMA(_Symbol, InpTimeframe, InpEmaFast, 0, MODE_EMA, PRICE_CLOSE);
   g_hEmaSlow = iMA(_Symbol, InpTimeframe, InpEmaSlow, 0, MODE_EMA, PRICE_CLOSE);
   g_hRsi     = iRSI(_Symbol, InpTimeframe, InpRsiPeriod, PRICE_CLOSE);
   g_hAtr     = iATR(_Symbol, InpTimeframe, 14);

   if(InpUseADX)
      g_hAdx = iADX(_Symbol, InpTimeframe, 14);
   if(InpUseMACD)
      g_hMacd = iMACD(_Symbol, InpTimeframe, 12, 26, 9, PRICE_CLOSE);

   if(g_hEmaFast == INVALID_HANDLE || g_hEmaSlow == INVALID_HANDLE ||
      g_hRsi == INVALID_HANDLE || g_hAtr == INVALID_HANDLE)
   {
      Print("[R10_MT5_EA] Erro ao criar handles principais.");
      return(INIT_FAILED);
   }

   if(InpUseADX && g_hAdx == INVALID_HANDLE)
   {
      Print("[R10_MT5_EA] Erro ao criar handle ADX.");
      return(INIT_FAILED);
   }

   if(InpUseMACD && g_hMacd == INVALID_HANDLE)
   {
      Print("[R10_MT5_EA] Erro ao criar handle MACD.");
      return(INIT_FAILED);
   }

   ResetState();
   LogMsg("Inicializado com sucesso.");
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   if(g_hEmaFast != INVALID_HANDLE) IndicatorRelease(g_hEmaFast);
   if(g_hEmaSlow != INVALID_HANDLE) IndicatorRelease(g_hEmaSlow);
   if(g_hRsi     != INVALID_HANDLE) IndicatorRelease(g_hRsi);
   if(g_hAtr     != INVALID_HANDLE) IndicatorRelease(g_hAtr);
   if(g_hAdx     != INVALID_HANDLE) IndicatorRelease(g_hAdx);
   if(g_hMacd    != INVALID_HANDLE) IndicatorRelease(g_hMacd);
}

void OnTick()
{
   datetime bar0 = iTime(_Symbol, InpTimeframe, 0);
   if(bar0 <= 0)
      return;

   if(bar0 == g_lastBarTime)
      return;

   g_lastBarTime = bar0;
   ProcessClosedBar();
}

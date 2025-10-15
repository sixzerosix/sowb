package market

import (
	"math"
)

// RSI Indicator Implementation
func NewRSIIndicator(period int) *RSIIndicator {
	return &RSIIndicator{
		period: period,
		gains:  make([]float64, 0, period),
		losses: make([]float64, 0, period),
	}
}

func (rsi *RSIIndicator) Update(price float64) {
	if rsi.lastPrice == 0 {
		rsi.lastPrice = price
		return
	}

	change := price - rsi.lastPrice

	if change > 0 {
		rsi.gains = append(rsi.gains, change)
		rsi.losses = append(rsi.losses, 0)
	} else {
		rsi.gains = append(rsi.gains, 0)
		rsi.losses = append(rsi.losses, -change)
	}

	// Ограничиваем размер массивов
	if len(rsi.gains) > rsi.period {
		rsi.gains = rsi.gains[1:]
		rsi.losses = rsi.losses[1:]
	}

	rsi.lastPrice = price
}

func (rsi *RSIIndicator) GetValue() float64 {
	if len(rsi.gains) < rsi.period {
		return 50 // Нейтральное значение
	}

	// Вычисляем средние значения
	avgGain := 0.0
	avgLoss := 0.0

	for i := 0; i < rsi.period; i++ {
		avgGain += rsi.gains[len(rsi.gains)-rsi.period+i]
		avgLoss += rsi.losses[len(rsi.losses)-rsi.period+i]
	}

	avgGain /= float64(rsi.period)
	avgLoss /= float64(rsi.period)

	if avgLoss == 0 {
		return 100
	}

	rs := avgGain / avgLoss
	return 100 - (100 / (1 + rs))
}

// EMA Indicator Implementation
func NewEMAIndicator(period int) *EMAIndicator {
	return &EMAIndicator{
		period:     period,
		multiplier: 2.0 / (float64(period) + 1.0),
	}
}

func (ema *EMAIndicator) Update(price float64) {
	if !ema.initialized {
		ema.ema = price
		ema.initialized = true
	} else {
		ema.ema = (price * ema.multiplier) + (ema.ema * (1 - ema.multiplier))
	}
}

func (ema *EMAIndicator) GetValue() float64 {
	return ema.ema
}

// MACD Indicator Implementation
func NewMACDIndicator() *MACDIndicator {
	return &MACDIndicator{
		ema12:  NewEMAIndicator(12),
		ema26:  NewEMAIndicator(26),
		signal: NewEMAIndicator(9),
	}
}

func (macd *MACDIndicator) Update(ema12, ema26 float64) {
	macdLine := ema12 - ema26
	macd.signal.Update(macdLine)
	macd.histogram = macdLine - macd.signal.GetValue()
}

func (macd *MACDIndicator) GetMACD() float64 {
	return macd.ema12.GetValue() - macd.ema26.GetValue()
}

func (macd *MACDIndicator) GetSignal() float64 {
	return macd.signal.GetValue()
}

func (macd *MACDIndicator) GetHistogram() float64 {
	return macd.histogram
}

// Bollinger Bands Implementation
func NewBollingerBandsIndicator(period int, deviation float64) *BollingerBandsIndicator {
	return &BollingerBandsIndicator{
		period:    period,
		deviation: deviation,
		prices:    make([]float64, 0, period),
	}
}

func (bb *BollingerBandsIndicator) Update(price float64) {
	bb.prices = append(bb.prices, price)

	if len(bb.prices) > bb.period {
		bb.prices = bb.prices[1:]
	}

	if len(bb.prices) >= bb.period {
		// Вычисляем SMA
		sum := 0.0
		for _, p := range bb.prices {
			sum += p
		}
		bb.sma = sum / float64(len(bb.prices))
		bb.middle = bb.sma

		// Вычисляем стандартное отклонение
		variance := 0.0
		for _, p := range bb.prices {
			variance += math.Pow(p-bb.sma, 2)
		}
		stdDev := math.Sqrt(variance / float64(len(bb.prices)))

		// Вычисляем полосы
		bb.upper = bb.middle + (bb.deviation * stdDev)
		bb.lower = bb.middle - (bb.deviation * stdDev)
	}
}

// Volume Profile Implementation
func NewVolumeProfileIndicator() *VolumeProfileIndicator {
	return &VolumeProfileIndicator{
		priceRanges: make(map[float64]float64),
	}
}

func (vp *VolumeProfileIndicator) Update(price, volume float64) {
	// Округляем цену до ближайшего тика
	roundedPrice := math.Round(price*100) / 100
	vp.priceRanges[roundedPrice] += volume

	// Находим Point of Control (цена с максимальным объемом)
	maxVolume := 0.0
	for p, v := range vp.priceRanges {
		if v > maxVolume {
			maxVolume = v
			vp.pocPrice = p
		}
	}
}

// Order Flow Delta Implementation
func NewOrderFlowDeltaIndicator() *OrderFlowDeltaIndicator {
	return &OrderFlowDeltaIndicator{}
}

func (ofd *OrderFlowDeltaIndicator) UpdateTrade(side string, volume float64) {
	if side == "Buy" {
		ofd.buyVolume += volume
	} else {
		ofd.sellVolume += volume
	}

	ofd.delta = ofd.buyVolume - ofd.sellVolume
	ofd.cumDelta += ofd.delta
}

func (ofd *OrderFlowDeltaIndicator) GetDelta() float64 {
	return ofd.delta
}

func (ofd *OrderFlowDeltaIndicator) GetCumulativeDelta() float64 {
	return ofd.cumDelta
}

func (ofd *OrderFlowDeltaIndicator) Reset() {
	ofd.buyVolume = 0
	ofd.sellVolume = 0
	ofd.delta = 0
}

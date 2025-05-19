interface TestDataProfile {
    trend: 'up' | 'down' | 'flat';
    volatility: 'volatile' | 'non-volatile';
}

interface TestDataOptions {
    length: number;
    profile1: TestDataProfile;
    profile2?: TestDataProfile;  // Optional second profile for full mode
    startDate?: Date;
}

export class TestDataGenerator {
    // Geometric Brownian Motion (GBM) for realistic stock prices
    private static generateGBM(
        length: number,
        startPrice: number,
        drift: number,
        volatility: number,
        jumpProb = 0.01,
        jumpScale = 0.04
    ): number[] {
        const dt = 1 / 252; // daily steps
        const prices = [startPrice];
        for (let i = 1; i < length; i++) {
            const Z = this.randn_bm();
            let jump = 0;
            if (Math.random() < jumpProb) {
                // Random jump (positive or negative)
                jump = (Math.random() - 0.5) * 2 * jumpScale;
            }
            const prev = prices[i - 1];
            // GBM formula with jump
            const next = prev * Math.exp((drift - 0.5 * volatility ** 2) * dt + volatility * Math.sqrt(dt) * Z + jump);
            prices.push(Math.max(0.01, next));
        }
        return prices;
    }

    // Standard normal random (Box-Muller)
    private static randn_bm(): number {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    public static generateTestData(options: TestDataOptions): { dates: string[], values: number[] } {
        const { length, profile1, profile2, startDate = new Date('1976-01-22') } = options;
        // Profile mapping
        const profileToParams = (profile: TestDataProfile) => {
            let drift = 0, sigma = 0.15;
            if (profile.trend === 'up') drift = 0.12;
            else if (profile.trend === 'down') drift = -0.12;
            else drift = 0.0;
            sigma = profile.volatility === 'volatile' ? 0.35 : 0.10;
            return { drift, sigma };
        };
        let finalData: number[];
        if (profile2) {
            // Two segments
            const mid = Math.floor(length / 2);
            const params1 = profileToParams(profile1);
            const params2 = profileToParams(profile2);
            const seg1 = this.generateGBM(mid, 100, params1.drift, params1.sigma);
            const seg2 = this.generateGBM(length - mid, 100, params2.drift, params2.sigma);
            // Shift seg2 so its first value matches the last value of seg1
            const offset = seg1[seg1.length - 1] - seg2[0];
            const seg2Shifted = seg2.map(val => val + offset);
            finalData = [...seg1, ...seg2Shifted];
        } else {
            const params = profileToParams(profile1);
            finalData = this.generateGBM(length, 100, params.drift, params.sigma);
        }
        // Generate dates
        const dates = Array.from({ length }, (_, i) => {
            const date = new Date(startDate);
            date.setDate(date.getDate() + i);
            return date.toISOString().split('T')[0];
        });
        return {
            dates,
            values: finalData.map(Math.abs)
        };
    }
} 
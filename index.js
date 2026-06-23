import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { GoogleGenerativeAI } from '@google/generative-ai';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

app.get('/', (req, res) => {
  res.send('Infinity Bulletproof Engine with Auto-Retry Layer Live! ⚡🚀');
});

app.post('/api/generate-test', async (req, res) => {
  try {
    const { exam, topic, count, type, difficulty, language } = req.body;

    if (!topic) return res.status(400).json({ error: "Topic missing bhai!" });

    const totalRequested = parseInt(count) || 5;
    const targetExam = exam || "Competitive Exam";
    const qType = type || "Objective";
    const diffLevel = difficulty || "Medium";
    const lang = language || "English";

    const model = genAI.getGenerativeModel({ 
      model: "gemini-2.5-flash",
      generationConfig: { responseMimeType: "application/json" }
    });

    // 🛑 CHUNKING LAYER: Max 15 questions per request to avoid token truncation
    const MAX_CHUNK_SIZE = 15;
    let allCompiledQuestions = [];
    let remainingQuestions = totalRequested;

    console.log(`🚀 Starting Safe Pipeline: Requested ${totalRequested} Qs for topic "${topic}"`);

    while (remainingQuestions > 0) {
      const currentChunkSize = Math.min(MAX_CHUNK_SIZE, remainingQuestions);
      let prompt = "";

      // 🔥 LASER-SHARP FOCUS + ULTRA-CONDENSED INPUT PROMPTS
      if (qType === 'Objective') {
        prompt = `Set ${currentChunkSize} direct, syllabus-precise MCQs for ${targetExam}. Focus ONLY on exact core topic: "${topic}". Difficulty: ${diffLevel}. Lang: ${lang}. No general fluff.
        JSON schema: {"questions": [{"id":0,"question":"","options":["","","",""],"correctOptionIndex":0,"explanation":""}]}.
        Explanation Rules (Max 20 words): GK/Sci/Eng=1-line direct core fact. Math=Only pure symbolic formula & 1-line execution step. No text stories.`;
      } else {
        prompt = `Set ${currentChunkSize} high-yield, specific analytical descriptive questions for ${targetExam}. Focus ONLY on exact core topic: "${topic}". Difficulty: ${diffLevel}. Lang: ${lang}. No generic talk. No options.
        JSON schema: {"questions": [{"id":0,"question":"","explanation":""}]}.
        Explanation Rules (Max 35 words): Give strict core valuation points or model answer outline framework.`;
      }

      // 🔄 AUTOMATED RETRY LAYER: Defends against Google 503 high demand spikes
      let retries = 3;
      let responseText = "";
      
      while (retries > 0) {
        try {
          console.log(`📦 Requesting batch of ${currentChunkSize} questions... (Attempts left: ${retries})`);
          const result = await model.generateContent(prompt);
          const response = await result.response;
          responseText = response.text();
          break; // Success! Break out of the retry loop instantly
        } catch (apiError) {
          retries--;
          console.warn(`⚠️ Google Server high demand spike hit. Retrying in 2 seconds...`);
          if (retries === 0) throw apiError; // If all 3 attempts fail, bubble up the error safely
          await new Promise(resolve => setTimeout(resolve, 2000)); // 2-second sleep state for server cool down
        }
      }

      // Sync and extract JSON structure
      const jsonMatch = responseText.match(/\{[\s\S]*\}/);
      if (!jsonMatch) throw new Error("Invalid AI JSON stream layer returned.");

      const parsedData = JSON.parse(jsonMatch[0]);
      if (parsedData.questions && Array.isArray(parsedData.questions)) {
        allCompiledQuestions = [...allCompiledQuestions, ...parsedData.questions];
      }

      remainingQuestions -= currentChunkSize;
    }

    // 🎯 CLEANUP LAYER: Re-index IDs sequentially from 0 to N for frontend mapping arrays
    const finalIndexedQuestions = allCompiledQuestions.map((q, index) => ({
      ...q,
      id: index
    }));

    console.log(`✅ Pipeline Success! Successfully merged ${finalIndexedQuestions.length} targeted questions.`);

    res.json({
      success: true,
      topic: topic,
      questions: finalIndexedQuestions
    });

  } catch (error) {
    console.error("❌ Final Backend Error Log:", error);
    res.status(500).json({ 
      success: false, 
      error: error.message || "Unknown locha occurred in pipeline module."
    });
  }
});

app.listen(PORT, () => console.log(`🔥 Server active on port: ${PORT}`));
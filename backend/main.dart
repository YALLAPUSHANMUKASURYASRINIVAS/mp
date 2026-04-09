import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:audioplayers/audioplayers.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

// ─── CONFIG ────────────────────────────────────────────────
// Change this to your Render/Railway URL after deploying backend
const String API_BASE = "https://mp-uulq.onrender.com";

// ─── MAIN ───────────────────────────────────────────────────
void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const OCRApp());
}

class OCRApp extends StatelessWidget {
  const OCRApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Telugu OCR Translator',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF6C63FF),
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
        fontFamily: 'Roboto',
      ),
      home: const HomeScreen(),
    );
  }
}

// ─── HOME SCREEN ────────────────────────────────────────────
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  // State
  File? _image;
  String _ocrText = "";
  String _translatedText = "";
  String _detectedLang = "";
  String _selectedLang = "en";
  bool _isLoading = false;
  bool _isPlaying = false;
  String _statusMessage = "";

  final ImagePicker _picker = ImagePicker();
  final AudioPlayer _audioPlayer = AudioPlayer();

  final Map<String, String> _languages = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Punjabi": "pa",
  };

  // ── Pick image ──────────────────────────────────────────
  Future<void> _pickImage(ImageSource source) async {
    final XFile? picked = await _picker.pickImage(
      source: source,
      imageQuality: 90,
    );
    if (picked == null) return;

    setState(() {
      _image = File(picked.path);
      _ocrText = "";
      _translatedText = "";
      _detectedLang = "";
      _statusMessage = "";
    });

    await _runOCRAndTranslate();
  }

  // ── OCR + Translate ─────────────────────────────────────
  Future<void> _runOCRAndTranslate() async {
    if (_image == null) return;

    setState(() {
      _isLoading = true;
      _statusMessage = "Scanning image... 🔍";
    });

    try {
      final uri = Uri.parse("$API_BASE/ocr-translate?target_language=$_selectedLang");
      final request = http.MultipartRequest("POST", uri);
      request.files.add(
        await http.MultipartFile.fromPath("file", _image!.path),
      );

      final streamed = await request.send().timeout(const Duration(seconds: 60));
      final response = await http.Response.fromStream(streamed);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _ocrText = data["ocr_text"] ?? "";
          _translatedText = data["translated"] ?? "";
          _detectedLang = data["detected_language"] ?? "";
          _statusMessage = "Done ✅";
        });
      } else {
        setState(() {
          _statusMessage = "Error: ${response.statusCode}";
        });
      }
    } catch (e) {
      setState(() {
        _statusMessage = "Connection error: $e";
      });
    } finally {
      setState(() => _isLoading = false);
    }
  }

  // ── Text to Speech ──────────────────────────────────────
  Future<void> _playTTS() async {
    if (_translatedText.isEmpty) return;

    setState(() => _isPlaying = true);

    try {
      final uri = Uri.parse("$API_BASE/tts");
      final response = await http.post(
        uri,
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "text": _translatedText,
          "target_language": _selectedLang,
        }),
      );

      if (response.statusCode == 200) {
        final tempDir = Directory.systemTemp;
        final file = File("${tempDir.path}/tts_output.mp3");
        await file.writeAsBytes(response.bodyBytes);
        await _audioPlayer.play(DeviceFileSource(file.path));
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("TTS Error: $e")),
      );
    } finally {
      setState(() => _isPlaying = false);
    }
  }

  // ── Retranslate when language changes ───────────────────
  Future<void> _retranslate() async {
    if (_ocrText.isEmpty) return;

    setState(() {
      _isLoading = true;
      _statusMessage = "Translating... 🌍";
    });

    try {
      final uri = Uri.parse("$API_BASE/translate");
      final response = await http.post(
        uri,
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "text": _ocrText,
          "target_language": _selectedLang,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _translatedText = data["translated"] ?? "";
          _statusMessage = "Done ✅";
        });
      }
    } catch (e) {
      setState(() => _statusMessage = "Error: $e");
    } finally {
      setState(() => _isLoading = false);
    }
  }

  // ── BUILD ───────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0F0F1A),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1A1A2E),
        title: const Row(
          children: [
            Text("🔤", style: TextStyle(fontSize: 22)),
            SizedBox(width: 8),
            Text(
              "Telugu OCR",
              style: TextStyle(
                color: Color(0xFF6C63FF),
                fontWeight: FontWeight.bold,
                fontSize: 20,
              ),
            ),
          ],
        ),
        elevation: 0,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // ── Image Preview ──────────────────────────
            _buildImageCard(),
            const SizedBox(height: 16),

            // ── Scan Buttons ───────────────────────────
            _buildScanButtons(),
            const SizedBox(height: 16),

            // ── Language Selector ──────────────────────
            _buildLanguageSelector(),
            const SizedBox(height: 16),

            // ── Loading / Status ───────────────────────
            if (_isLoading) _buildLoadingCard(),

            // ── OCR Result ─────────────────────────────
            if (_ocrText.isNotEmpty) _buildOCRCard(),
            if (_ocrText.isNotEmpty) const SizedBox(height: 12),

            // ── Translation Result ─────────────────────
            if (_translatedText.isNotEmpty) _buildTranslationCard(),
          ],
        ),
      ),
    );
  }

  Widget _buildImageCard() {
    return Container(
      height: 220,
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A2E),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFF6C63FF).withOpacity(0.4)),
      ),
      child: _image == null
          ? const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.image_outlined, size: 60, color: Color(0xFF6C63FF)),
                  SizedBox(height: 8),
                  Text(
                    "No image selected",
                    style: TextStyle(color: Colors.white54),
                  ),
                ],
              ),
            )
          : ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: Image.file(_image!, fit: BoxFit.cover),
            ),
    );
  }

  Widget _buildScanButtons() {
    return Row(
      children: [
        Expanded(
          child: _GradientButton(
            icon: Icons.photo_library,
            label: "Gallery",
            onPressed: () => _pickImage(ImageSource.gallery),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: _GradientButton(
            icon: Icons.camera_alt,
            label: "Camera",
            onPressed: () => _pickImage(ImageSource.camera),
          ),
        ),
      ],
    );
  }

  Widget _buildLanguageSelector() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A2E),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFF6C63FF).withOpacity(0.4)),
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<String>(
          value: _selectedLang,
          isExpanded: true,
          dropdownColor: const Color(0xFF1A1A2E),
          style: const TextStyle(color: Colors.white, fontSize: 16),
          icon: const Icon(Icons.language, color: Color(0xFF6C63FF)),
          hint: const Text("🌍 Translate to..."),
          items: _languages.entries
              .map((e) => DropdownMenuItem(
                    value: e.value,
                    child: Text("🌐 ${e.key}"),
                  ))
              .toList(),
          onChanged: (val) {
            if (val == null) return;
            setState(() => _selectedLang = val);
            _retranslate();
          },
        ),
      ),
    );
  }

  Widget _buildLoadingCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A2E),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          const SizedBox(
            width: 20, height: 20,
            child: CircularProgressIndicator(
              strokeWidth: 2,
              color: Color(0xFF6C63FF),
            ),
          ),
          const SizedBox(width: 12),
          Text(_statusMessage, style: const TextStyle(color: Colors.white70)),
        ],
      ),
    );
  }

  Widget _buildOCRCard() {
    return _ResultCard(
      icon: "📄",
      title: "Extracted Text",
      subtitle: _detectedLang.isNotEmpty ? "Detected: $_detectedLang" : null,
      content: _ocrText,
      color: const Color(0xFF6C63FF),
    );
  }

  Widget _buildTranslationCard() {
    return _ResultCard(
      icon: "🌍",
      title: "Translation",
      content: _translatedText,
      color: const Color(0xFF00BFA5),
      trailing: IconButton(
        onPressed: _playTTS,
        icon: _isPlaying
            ? const SizedBox(
                width: 20, height: 20,
                child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF00BFA5)),
              )
            : const Icon(Icons.volume_up, color: Color(0xFF00BFA5)),
        tooltip: "Play Audio",
      ),
    );
  }
}

// ─── REUSABLE WIDGETS ────────────────────────────────────────

class _GradientButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onPressed;

  const _GradientButton({
    required this.icon,
    required this.label,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onPressed,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 14),
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            colors: [Color(0xFF6C63FF), Color(0xFF3F3D9C)],
          ),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: Colors.white, size: 20),
            const SizedBox(width: 8),
            Text(
              label,
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                fontSize: 15,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  final String icon;
  final String title;
  final String? subtitle;
  final String content;
  final Color color;
  final Widget? trailing;

  const _ResultCard({
    required this.icon,
    required this.title,
    required this.content,
    required this.color,
    this.subtitle,
    this.trailing,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A2E),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(0.5)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(icon, style: const TextStyle(fontSize: 18)),
              const SizedBox(width: 8),
              Text(
                title,
                style: TextStyle(
                  color: color,
                  fontWeight: FontWeight.bold,
                  fontSize: 14,
                ),
              ),
              if (subtitle != null) ...[
                const SizedBox(width: 8),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(
                    color: color.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    subtitle!,
                    style: TextStyle(color: color, fontSize: 11),
                  ),
                ),
              ],
              const Spacer(),
              if (trailing != null) trailing!,
            ],
          ),
          const SizedBox(height: 10),
          SelectableText(
            content,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 16,
              height: 1.5,
            ),
          ),
        ],
      ),
    );
  }
}

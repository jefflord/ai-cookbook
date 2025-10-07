from youtube_transcript_api import YouTubeTranscriptApi


class YouTubeTranscriptFetcher:
    def __init__(self):
        self.api = YouTubeTranscriptApi()

    def get_transcript(self, video_id, languages=None):
        if languages is None:
            languages = ["en"]
        try:
            transcript = self.api.fetch(video_id, languages=languages)
            transcript_data = transcript.to_raw_data()
            return {
                "success": True,
                "data": {
                    "video_id": video_id,
                    "language": transcript.language,
                    "language_code": transcript.language_code,
                    "is_generated": transcript.is_generated,
                    "transcript": transcript_data,
                },
                "error": None,
            }
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}

    def get_transcript_text_only(self, video_id, languages=None):
        result = self.get_transcript(video_id, languages)
        if result["success"]:
            return " ".join(
                [snippet["text"] for snippet in result["data"]["transcript"]]
            )
        else:
            return f"Error: {result['error']}"


if __name__ == "__main__":
    fetcher = YouTubeTranscriptFetcher()
    transcript = fetcher.get_transcript_text_only("3SVksBB3_YY")
    print(transcript)

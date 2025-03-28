import cv2
import numpy as np
import pytesseract
import os

class AnimeWallpaperBot:
    def __init__(self, video_path, output_dir='wallpapers'):
        """
        Initialize the bot with video path and output directory
        
        Args:
            video_path (str): Path to the anime movie video file
            output_dir (str): Directory to save wallpaper frames
        """
        self.video_path = video_path
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load anime face detection cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def get_sharpness(self, frame):
        """
        Compute image sharpness using Laplacian variance
        
        Args:
            frame (numpy.ndarray): Input image frame
        
        Returns:
            float: Sharpness score
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def get_color_variance(self, frame):
        """
        Compute color variance in HSV color space
        
        Args:
            frame (numpy.ndarray): Input image frame
        
        Returns:
            float: Color variance score
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        return hue.var()

    def get_edge_density(self, frame):
        """
        Compute edge density using Canny edge detection
        
        Args:
            frame (numpy.ndarray): Input image frame
        
        Returns:
            float: Edge density score
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return edges.sum() / (edges.shape[0] * edges.shape[1])

    def get_symmetry(self, frame):
        """
        Compute frame symmetry by comparing left and right halves
        
        Args:
            frame (numpy.ndarray): Input image frame
        
        Returns:
            float: Symmetry score
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        left = gray[:, :gray.shape[1]//2]
        right = cv2.flip(gray[:, gray.shape[1]//2:], 1)
        diff = ((left - right) ** 2).mean()
        return 1 / (1 + diff)

    def detect_faces(self, frame):
        """
        Detect faces in the frame
        
        Args:
            frame (numpy.ndarray): Input image frame
        
        Returns:
            tuple: Number of faces and list of face sizes
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        return len(faces), [w * h for (x, y, w, h) in faces]

    def has_text(self, frame):
        """
        Check if frame contains text
        
        Args:
            frame (numpy.ndarray): Input image frame
        
        Returns:
            bool: True if text is detected, False otherwise
        """
        try:
            text = pytesseract.image_to_string(frame)
            return len(text.strip()) > 0
        except:
            return False

    def compute_beauty_score(self, frame):
        """
        Compute overall beauty score for a frame
        
        Args:
            frame (numpy.ndarray): Input image frame
        
        Returns:
            float: Beauty score
        """
        # Skip frames with text or low sharpness
        if self.has_text(frame) or self.get_sharpness(frame) < 100:
            return -1

        num_faces, face_sizes = self.detect_faces(frame)
        num_large_faces = sum(1 for size in face_sizes if size > 50000)

        # Weights for different features
        w_faces = 0.5
        w_color = 0.3
        w_edge = 0.1
        w_symmetry = 0.1

        score = (w_faces * num_large_faces) + \
                (w_color * self.get_color_variance(frame)) + \
                (w_edge * self.get_edge_density(frame)) + \
                (w_symmetry * self.get_symmetry(frame))

        return score

    def extract_wallpapers(self, num_wallpapers=10):
        """
        Extract wallpaper-worthy frames from the video
        
        Args:
            num_wallpapers (int): Number of top wallpapers to extract
        
        Returns:
            list: List of tuples (frame, score)
        """
        video = cv2.VideoCapture(self.video_path)
        frames_with_scores = []
        count = 0

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Extract 1 frame per second
            if count % int(video.get(cv2.CAP_PROP_FPS)) == 0:
                score = self.compute_beauty_score(frame)
                if score > 0:
                    frames_with_scores.append((frame, score))

            count += 1

        video.release()

        # Sort frames by beauty score and select top wallpapers
        frames_with_scores.sort(key=lambda x: x[1], reverse=True)
        top_frames = frames_with_scores[:num_wallpapers]

        # Save wallpapers
        for i, (frame, score) in enumerate(top_frames):
            filename = os.path.join(self.output_dir, f'wallpaper_{i}_score_{score:.2f}.png')
            cv2.imwrite(filename, frame)
            print(f"Saved wallpaper: {filename}")

        return top_frames

def main():
    # Example usage
    video_path = 'anime_movie.mp4'
    bot = AnimeWallpaperBot(video_path)
    bot.extract_wallpapers(num_wallpapers=10)

if __name__ == "__main__":
    main()

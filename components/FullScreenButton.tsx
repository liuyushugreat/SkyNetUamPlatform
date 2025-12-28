import React, { useState } from 'react';
import { Maximize2, Minimize2 } from 'lucide-react';

interface FullScreenButtonProps {
  className?: string;
  iconColor?: string;
}

const FullScreenButton: React.FC<FullScreenButtonProps> = ({ className, iconColor = 'currentColor' }) => {
  const [isFullScreen, setIsFullScreen] = useState(false);

  const toggleFullScreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().then(() => {
        setIsFullScreen(true);
      }).catch((err) => {
        console.error(`Error attempting to enable full-screen mode: ${err.message} (${err.name})`);
      });
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen().then(() => {
          setIsFullScreen(false);
        });
      }
    }
  };

  return (
    <button
      onClick={toggleFullScreen}
      className={`p-2 rounded-lg transition-colors hover:bg-opacity-20 hover:bg-gray-500 ${className}`}
      title={isFullScreen ? "Exit Fullscreen" : "Enter Fullscreen"}
    >
      {isFullScreen ? (
        <Minimize2 size={20} color={iconColor} />
      ) : (
        <Maximize2 size={20} color={iconColor} />
      )}
    </button>
  );
};

export default FullScreenButton;


import { Composition, continueRender, delayRender, staticFile } from "remotion";
import {
  CaptionedVideo,
  calculateCaptionedVideoMetadata,
  captionedVideoSchema,
} from "./CaptionedVideo";
import { getVideoMetadata } from "@remotion/media-utils";
import React, { useEffect, useState } from "react";
import { getInputProps } from "remotion";


export const RemotionRoot: React.FC = () => {
  const [handle] = useState(() => delayRender());
  const [width, setWidth] = React.useState<number>(0);
  const [height, setHeight] = React.useState<number>(0);
  const inputProps = getInputProps()
  const file = staticFile(inputProps.video as string);

  useEffect(() => {
    getVideoMetadata(
      file,
    )
      .then(({ width, height }) => {
        setWidth(width);
        setHeight(height);
        continueRender(handle);
      })
      .catch((err) => {
        console.log(`Error fetching metadata: ${err}`);
      });
  }, [handle]);

  return (
    <Composition
      id="CaptionedVideo"
      component={CaptionedVideo}
      calculateMetadata={calculateCaptionedVideoMetadata}
      schema={captionedVideoSchema}
      width={width}
      height={height}
      defaultProps={{
        src: file,
      }}
    />
  );
};

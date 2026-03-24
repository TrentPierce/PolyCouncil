import QtQuick
import QtQuick.Controls

Label {
    color: Theme.textMuted
    font.pixelSize: Qt.application.font.pixelSize ? Qt.application.font.pixelSize * 0.85 : 11
    font.letterSpacing: 2.2
    font.capitalization: Font.AllUppercase
}

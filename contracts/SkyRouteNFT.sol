// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title SkyRouteNFT
 * @dev Represents a Time-Space Slot (Right-of-Way) as a Non-Fungible Asset.
 *      Adheres to SkyNet-RWA-Nexus protocol.
 * 
 * Academic Concept:
 *      Each token $T_i$ represents a tuple $(x, y, z, t_{start}, t_{end})$.
 *      Ownership grants the exclusive right to occupy that voxel during that interval.
 */

contract SkyRouteNFT {
    
    string public name = "SkyNet Route Rights";
    string public symbol = "SKYR";
    
    // Simulating ERC721 basics for this prototype
    mapping(uint256 => address) private _owners;
    mapping(uint256 => RouteMetadata) public _routeData;
    uint256 private _tokenIds;

    struct RouteMetadata {
        string voxelId;     // Geo-spatial ID
        uint256 startTime;  // Unix Timestamp
        uint256 endTime;    // Unix Timestamp
        uint256 pricePaid;  // In SkyToken
        bool isVerified;    // Proof-of-Flight status
    }

    event RouteMinted(uint256 indexed tokenId, address indexed pilot, string voxelId);
    event FlightVerified(uint256 indexed tokenId, bool success);

    /**
     * @dev Mint a new Route Right via Auction Settlement.
     *      Must be called by the AuctionController contract.
     */
    function mintRoute(
        address to, 
        string memory voxelId, 
        uint256 startTime, 
        uint256 duration
    ) public returns (uint256) {
        _tokenIds++;
        uint256 newItemId = _tokenIds;
        
        _owners[newItemId] = to;
        
        _routeData[newItemId] = RouteMetadata({
            voxelId: voxelId,
            startTime: startTime,
            endTime: startTime + duration,
            pricePaid: 0, // Set by payment logic
            isVerified: false
        });

        emit RouteMinted(newItemId, to, voxelId);
        return newItemId;
    }

    /**
     * @dev Oracle Callback.
     *      SkyNet simulation engine calls this after t_end to verify
     *      if the pilot actually flew the route safely.
     */
    function verifyFlightLog(uint256 tokenId, bool safeFlight) public {
        // In prod: require(msg.sender == ORACLE_ADDRESS);
        RouteMetadata storage route = _routeData[tokenId];
        route.isVerified = safeFlight;
        
        emit FlightVerified(tokenId, safeFlight);
        
        // If !safeFlight, logic to slash stake would trigger here.
    }

    function ownerOf(uint256 tokenId) public view returns (address) {
        return _owners[tokenId];
    }
}


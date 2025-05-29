#pragma once
#include "gpu.hpp"
#include <solutionInterface/gpu.hpp>
#include <glm/glm.hpp>
#include <algorithm>
#include <vector>
#include <cstdint>

//-----------------------------------------------------------------------------
// Вспомогательные функции
//----------------------------------------------------------------------------- 
static uint32_t getVertexIndex(const GPUMemory& mem, uint32_t vid) {
    auto const& vao = mem.vertexArrays[mem.activatedVertexArray];
    if (vao.indexBufferID < 0) return vid;
    auto const& ib = mem.buffers[vao.indexBufferID];
    auto const* base = reinterpret_cast<uint8_t const*>(ib.data) + vao.indexOffset;
    switch (vao.indexType) {
    case IndexType::U8:  return base[vid];
    case IndexType::U16: return reinterpret_cast<uint16_t const*>(base)[vid];
    case IndexType::U32: return reinterpret_cast<uint32_t const*>(base)[vid];
    default:             return vid;
    }
}

static void fetchAttrib(
    Attrib& dst,
    Buffer const& buf,
    uint64_t      offset,
    uint64_t      stride,
    uint32_t      idx,
    AttribType    type)
{
    auto const* ptr = reinterpret_cast<uint8_t const*>(buf.data) + offset + stride * idx;
    switch (type) {
    case AttribType::FLOAT:  dst.v1 = *reinterpret_cast<float const*>(ptr); break;
    case AttribType::VEC2:   dst.v2 = *reinterpret_cast<glm::vec2 const*>(ptr); break;
    case AttribType::VEC3:   dst.v3 = *reinterpret_cast<glm::vec3 const*>(ptr); break;
    case AttribType::VEC4:   dst.v4 = *reinterpret_cast<glm::vec4 const*>(ptr); break;
    case AttribType::UINT:   dst.u1 = *reinterpret_cast<uint32_t const*>(ptr); break;
    case AttribType::UVEC2:  dst.u2 = *reinterpret_cast<glm::uvec2 const*>(ptr); break;
    case AttribType::UVEC3:  dst.u3 = *reinterpret_cast<glm::uvec3 const*>(ptr); break;
    case AttribType::UVEC4:  dst.u4 = *reinterpret_cast<glm::uvec4 const*>(ptr); break;
    default: break;
    }
}

//-----------------------------------------------------------------------------
// Очистка буферов
//----------------------------------------------------------------------------- 
static void clearColorBuffer(Framebuffer& fbo, ClearColorCommand const& cmd) {
    if (!fbo.color.data) return;
    for (uint32_t y = 0; y < fbo.height; ++y)
        for (uint32_t x = 0; x < fbo.width; ++x) {
            void* p = getPixel(fbo.color, x, y);
            if (fbo.color.format == Image::Format::U8) {
                auto* dst = reinterpret_cast<uint8_t*>(p);
                for (uint32_t c = 0; c < fbo.color.channels; ++c) {
                    float v = std::clamp(cmd.value[c], 0.f, 1.f);
                    dst[c] = uint8_t(v * 255.f + 0.5f);
                }
            }
            else {
                auto* dst = reinterpret_cast<float*>(p);
                for (uint32_t c = 0; c < fbo.color.channels; ++c)
                    dst[c] = cmd.value[c];
            }
        }
}
static void clearDepthBuffer(Framebuffer& fbo, ClearDepthCommand const& cmd) {
    if (!fbo.depth.data) return;
    for (uint32_t y = 0; y < fbo.height; ++y)
        for (uint32_t x = 0; x < fbo.width; ++x)
            *reinterpret_cast<float*>(getPixel(fbo.depth, x, y)) = cmd.value;
}
static void clearStencilBuffer(Framebuffer& fbo, ClearStencilCommand const& cmd) {
    if (!fbo.stencil.data) return;
    for (uint32_t y = 0; y < fbo.height; ++y)
        for (uint32_t x = 0; x < fbo.width; ++x)
            *reinterpret_cast<uint8_t*>(getPixel(fbo.stencil, x, y)) = cmd.value;
}
static void applyStencilOp(uint8_t* stencilValue, StencilOp op, uint32_t refValue) {
    switch (op) {
    case StencilOp::KEEP: break;
    case StencilOp::ZERO: *stencilValue = 0; break;
    case StencilOp::REPLACE: *stencilValue = uint8_t(refValue); break;
    case StencilOp::INCR: *stencilValue = uint8_t(std::min<uint32_t>(*stencilValue + 1, 255)); break;
    case StencilOp::INCR_WRAP: *stencilValue = uint8_t((*stencilValue + 1) & 0xFF); break;
    case StencilOp::DECR: *stencilValue = uint8_t(std::max<int>(*stencilValue - 1, 0)); break;
    case StencilOp::DECR_WRAP: *stencilValue = uint8_t((*stencilValue - 1) & 0xFF); break;
    case StencilOp::INVERT: *stencilValue = ~*stencilValue; break;
    }
}

struct VertexCache { std::vector<OutVertex> verts; };

//-----------------------------------------------------------------------------
// Основная обработка команд и pipeline
//----------------------------------------------------------------------------- 
static void processCommands(GPUMemory& mem, CommandBuffer const& cb) {
    for (uint32_t ci = 0; ci < cb.nofCommands; ++ci) {
        auto& cmd = cb.commands[ci];
        switch (cmd.type) {
        case CommandType::BIND_FRAMEBUFFER:
            mem.activatedFramebuffer = cmd.data.bindFramebufferCommand.id;
            break;
        case CommandType::BIND_PROGRAM:
            mem.activatedProgram = cmd.data.bindProgramCommand.id;
            break;
        case CommandType::BIND_VERTEXARRAY:
            mem.activatedVertexArray = cmd.data.bindVertexArrayCommand.id;
            break;
        case CommandType::BLOCK_WRITES_COMMAND:
            mem.blockWrites = cmd.data.blockWritesCommand.blockWrites;
            break;
        case CommandType::SET_BACKFACE_CULLING_COMMAND:
            mem.backfaceCulling.enabled = cmd.data.setBackfaceCullingCommand.enabled;
            break;
        case CommandType::SET_FRONT_FACE_COMMAND:
            mem.backfaceCulling.frontFaceIsCounterClockWise = cmd.data.setFrontFaceCommand.frontFaceIsCounterClockWise;
            break;
        case CommandType::SET_STENCIL_COMMAND:
            mem.stencilSettings = cmd.data.setStencilCommand.settings;
            break;
        case CommandType::SET_DRAW_ID:
            mem.gl_DrawID = cmd.data.setDrawIdCommand.id;
            break;
        case CommandType::CLEAR_COLOR:
            clearColorBuffer(mem.framebuffers[mem.activatedFramebuffer], cmd.data.clearColorCommand);
            break;
        case CommandType::CLEAR_DEPTH:
            clearDepthBuffer(mem.framebuffers[mem.activatedFramebuffer], cmd.data.clearDepthCommand);
            break;
        case CommandType::CLEAR_STENCIL:
            clearStencilBuffer(mem.framebuffers[mem.activatedFramebuffer], cmd.data.clearStencilCommand);
            break;
        case CommandType::USER_COMMAND:
            if (cmd.data.userCommand.callback)
                cmd.data.userCommand.callback(cmd.data.userCommand.data);
            break;
        case CommandType::DRAW: {
            auto& draw = cmd.data.drawCommand;
            auto& prog = mem.programs[mem.activatedProgram];
            ShaderInterface si_vs{ mem.uniforms,mem.textures,mem.gl_DrawID };
            // Vertex Shading
            VertexCache cache; cache.verts.reserve(draw.nofVertices);
            auto const& vao = mem.vertexArrays[mem.activatedVertexArray];
            for (uint32_t v = 0; v < draw.nofVertices; ++v) {
                uint32_t idx = getVertexIndex(mem, v);
                InVertex in{ }; in.gl_VertexID = idx;
                for (int a = 0;a < maxAttribs;++a) {
                    auto const& va = vao.vertexAttrib[a];
                    if (va.bufferID < 0 || va.type == AttribType::EMPTY) continue;
                    fetchAttrib(in.attributes[a], mem.buffers[va.bufferID], va.offset, va.stride, idx, va.type);
                }
                OutVertex out{ };
                if (prog.vertexShader) prog.vertexShader(out, in, si_vs);
                cache.verts.push_back(out);
            }
            // Viewport
            auto& fb = mem.framebuffers[mem.activatedFramebuffer];
            float W = float(fb.width), H = float(fb.height) * (fb.yReversed ? -1.f : 1.f);
            std::vector<glm::vec4> scr; scr.reserve(cache.verts.size());
            for (auto& o : cache.verts) {
                glm::vec4 p = o.gl_Position / o.gl_Position.w;
                scr.push_back({ (p.x * 0.5f + 0.5f) * W,(p.y * 0.5f + 0.5f) * H,p.z,o.gl_Position.w });
            }
            // Raster
            auto edgeF = [&](const glm::vec4& P0, const glm::vec4& P1, float px, float py) {
                return (px - P0.x) * (P1.y - P0.y) - (py - P0.y) * (P1.x - P0.x);
                };
            auto isTopLeft = [&](const glm::vec4& P0, const glm::vec4& P1) {
                return (P0.y > P1.y) || (P0.y == P1.y && P0.x < P1.x);
                };
            size_t primCount = cache.verts.size() / 3;
            for (size_t ti = 0;ti < primCount;++ti) {
                size_t b = ti * 3;
                auto A_pos = scr[b], B_pos = scr[b + 1], C_pos = scr[b + 2];
                auto A_v = cache.verts[b], B_v = cache.verts[b + 1], C_v = cache.verts[b + 2];
                float area2 = edgeF(A_pos, B_pos, C_pos.x, C_pos.y);
                if (std::abs(area2) < 1e-9f) continue;
                if (area2 < 0) { std::swap(B_pos, C_pos);std::swap(B_v, C_v);area2 = -area2; }
                int minX = std::max(0, int(std::floor(std::min({ A_pos.x,B_pos.x,C_pos.x }))));
                int maxX = std::min(int(fb.width) - 1, int(std::ceil(std::max({ A_pos.x,B_pos.x,C_pos.x }))));
                int minY = std::max(0, int(std::floor(std::min({ A_pos.y,B_pos.y,C_pos.y }))));
                int maxY = std::min(int(fb.height) - 1, int(std::ceil(std::max({ A_pos.y,B_pos.y,C_pos.y }))));
                float invA = 1.f / area2;
                for (int py = minY;py <= maxY;++py)for (int px = minX;px <= maxX;++px) {
                    float fx = px + 0.5f, fy = py + 0.5f;
                    float f0 = edgeF(A_pos, B_pos, fx, fy);
                    float f1 = edgeF(B_pos, C_pos, fx, fy);
                    float f2 = edgeF(C_pos, A_pos, fx, fy);
                    if (f0 < 0 || f1 < 0 || f2 < 0) continue;
                    if (f0 == 0 && !isTopLeft(A_pos, B_pos)) continue;
                    if (f1 == 0 && !isTopLeft(B_pos, C_pos)) continue;
                    if (f2 == 0 && !isTopLeft(C_pos, A_pos)) continue;
                    float w0 = f1 * invA;  // weight for vertex A
                    float w1 = f2 * invA;  // weight for vertex B
                    float w2 = f0 * invA;  // weight for vertex C
                    // depth test
                    bool passDepth = true;
                    float z = w0 * A_pos.z + w1 * B_pos.z + w2 * C_pos.z;
                    if (fb.depth.data) {
                        float* db = reinterpret_cast<float*>(getPixel(fb.depth, px, py));
                        if (z > *db) passDepth = false;
                        else if (!mem.blockWrites.depth) *db = z;
                    }
                    if (!passDepth) continue;
                    // fragment shader
                    InFragment inF;
                    inF.gl_FragCoord = { fx,fy,z,1 };
                    for (int a = 0;a < maxAttribs;++a) {
                        if (prog.vs2fs[a] == AttribType::EMPTY) continue;
                        if (prog.vs2fs[a] == AttribType::UINT || prog.vs2fs[a] == AttribType::UVEC2 ||
                            prog.vs2fs[a] == AttribType::UVEC3 || prog.vs2fs[a] == AttribType::UVEC4) {
                            inF.attributes[a] = A_v.attributes[a];
                        }
                        else {
                            float h0 = A_v.gl_Position.w, h1 = B_v.gl_Position.w, h2 = C_v.gl_Position.w;
                            float q0 = w0 / h0, q1 = w1 / h1, q2 = w2 / h2; float s = 1.f / (q0 + q1 + q2); q0 *= s; q1 *= s; q2 *= s;
                            inF.attributes[a].v4 = A_v.attributes[a].v4 * q0 + B_v.attributes[a].v4 * q1 + C_v.attributes[a].v4 * q2;
                        }
                    }
                    OutFragment outF{};
                    ShaderInterface si_fs{ mem.uniforms,mem.textures,mem.gl_DrawID };
                    if (prog.fragmentShader) prog.fragmentShader(outF, inF, si_fs);
                    if (outF.discard) continue;
                    // color write
                    if (fb.color.data && !mem.blockWrites.color) {
                        void* dp = getPixel(fb.color, px, py);
                        if (fb.color.format == Image::Format::U8) {
                            auto* dst = reinterpret_cast<uint8_t*>(dp);
                            float alpha = std::clamp(outF.gl_FragColor.a, 0.f, 1.f);
                            for (uint32_t c = 0;c < std::min(fb.color.channels, 3u);++c) {
                                float curr = float(dst[c]) / 255.f;
                                float col = std::clamp(outF.gl_FragColor[c], 0.f, 1.f);
                                float blended = curr * (1 - alpha) + col * alpha;
                                dst[c] = uint8_t(blended * 255.f + 0.5f);
                            }
                            if (fb.color.channels >= 4) dst[3] = uint8_t(std::clamp(outF.gl_FragColor.a, 0.f, 1.f) * 255.f + 0.5f);
                        }
                        else {
                            auto* dst = reinterpret_cast<float*>(dp);
                            float alpha = outF.gl_FragColor.a;
                            for (uint32_t c = 0;c < std::min(fb.color.channels, 3u);++c)
                                dst[c] = dst[c] * (1 - alpha) + outF.gl_FragColor[c] * alpha;
                            if (fb.color.channels >= 4) dst[3] = outF.gl_FragColor.a;
                        }
                    }
                }
            }
            ++mem.gl_DrawID;
            break;
        }
        case CommandType::SUB_COMMAND:
            if (cmd.data.subCommand.commandBuffer)
                processCommands(mem, *cmd.data.subCommand.commandBuffer);
            break;
        default: break;
        }
    }
}

void student_GPU_run(GPUMemory& mem, CommandBuffer const& cb) {
    mem.activatedFramebuffer = mem.defaultFramebuffer;
    mem.gl_DrawID = 0;
    processCommands(mem, cb);
}
